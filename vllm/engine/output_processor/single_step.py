from typing import Dict, List, Optional, Tuple, Union

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, Sequence, SequenceGroup, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)


def single_step_process_prompt_logprob(
        sg_output_proc: SequenceGroupOutputProcessor, seq_group: SequenceGroup,
        output: SequenceGroupOutput) -> None:
    """Process prompt logprobs associated with the :class:`SequenceGroupOutput`
    for a given step.

    Do nothing if the output has no prompt logprobs.

    Account for the fact that transformers do not compute first-token logprobs.
    
    Args:
      sg_output_proc: :class:`SequenceGroupOutputProcessor` instance
      seq_group: the output is associated with this :class:`SequenceGroup`
      output: the :class:`SequenceGroupOutput` for a single scheduler step
    """
    # 提示词token的对数概率列表
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = output.prompt_logprobs

    # If this is the first (or only) "chunk" of the prefill, we need
    # to prepend None to the list of prompt logprobs. The reason for this
    # is that for N prompt tokens, the Sampler will generate N-1 total
    # prompt logprobs during prefill since the token at idx 0 will not
    # have a logprob associated with it.
    if prompt_logprobs is not None:
        if not seq_group.prompt_logprobs:
            # 索引为0的提示词token没有对数概率,这里填充上
            prompt_logprobs = [None] + prompt_logprobs
            seq_group.prompt_logprobs = []

        assert hasattr(sg_output_proc, 'detokenizer')
        if (seq_group.sampling_params.detokenize
                and sg_output_proc.detokenizer):
            # 根据提示词对数概率进行就地解码(存到prompt_logprobs中Logprob.decoded_token字段)
            sg_output_proc.detokenizer.decode_prompt_logprobs_inplace(
                seq_group,
                prompt_logprobs,
                position_offset=len(seq_group.prompt_logprobs))

        seq_group.prompt_logprobs.extend(prompt_logprobs)


class SingleStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles "output processing" logic,
    which happens after the model returns generated token ids and before
    scheduling of the next batch. Output processing logic includes
    detokenization, and determining if a sequence is finished (e.g. via max len
    or eos token).

    The SingleStepOutputProcessor is specialized to the case where the model
    emits at most a single token per invocation, which precludes configurations
    such as speculative decoding or multi-step decoding. This enables beam
    search sampling, which requires forking/finishing/freeing sequences in a way
    that is currently difficult to schedule multiple steps ahead of time.
    """

    def __init__(self, scheduler_config: SchedulerConfig,
                 detokenizer: Detokenizer, scheduler: List[Scheduler],
                 seq_counter: Counter, stop_checker: StopChecker):
        self.scheduler_config = scheduler_config
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        """每个PP-stage的调度器"""
        self.seq_counter = seq_counter
        """序列ID计数器"""
        self.stop_checker = stop_checker

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput],
                        is_async: bool) -> None:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        
        is_async - Indicates whether this postprocessor runs in 
            parallel with the GPU forward pass and is processing 
            tokens from the previous step. If this is true, then
            no tokens need to be appended since it is already done
            externally (before the next schedule() call)
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0],
                                                    is_async)

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        """Process prompt logprobs associated with one step of a single-step-
        scheduled computation.
        
        Args:
          seq_group: the output is associated with this :class:`SequenceGroup`
          output: the :class:`SequenceGroupOutput` for a single scheduler step
        """
        assert len(outputs) == 1, ("Single step should only has 1 output.")
        output: SequenceGroupOutput = outputs[0]
        single_step_process_prompt_logprob(self, seq_group, output)

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput,
                                        is_async: bool) -> None:
        """基于多输出序列/Beam-search等规则处理推理输出token到序列分组并进行序列的KV-Cache分块表的增删"""
        sampling_params = seq_group.sampling_params
        # 单个生成序列且不使用beam-search
        if sampling_params.best_of == 1 and not sampling_params.use_beam_search:
            # only have one output sample
            sample: SequenceOutput = outputs.samples[0]
            # only have one sequence
            seq: Sequence = seq_group.seqs[0]
            # 不异步处理输出
            if not is_async:
                # 添加输出token到序列
                seq.append_token_id(sample.output_token, sample.logprobs)
            if sampling_params.detokenize and self.detokenizer:
                # 就地解码序列新增的输出token
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, sampling_params)
            else:
                new_char_count = 0
            # 判断序列生成是否终止
            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count,
                sampling_params,
                lora_req=seq_group.lora_request,
            )
            if seq.is_finished():
                # 遍历每个调度器释放该序列的KV-Cache分块表
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)
            return
        
        # 生成多个序列或使用beam-search 

        # TODO: Add support for async for beam search
        # 当前不支持异步处理输出
        assert not is_async

        # Process samples
        samples: List[SequenceOutput] = outputs.samples
        # 运行中的序列(作为父序列)
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        # 父序列中已完成的序列列表
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict: Dict[int, List[SequenceOutput]] = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            # Guard against a KeyError which can occur if the request was
            # aborted while the output was generated
            # 添加采样token到相应的父序列的子序列列表
            if (child_list :=   # 父序列的子序列列表
                    parent_child_dict.get(sample.parent_seq_id)) is not None:
                child_list.append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                # 该父序列没有新生成的token追加,需要被移除
                parent.status = SequenceStatus.FINISHED_ABORTED
                # 移除该序列
                seq_group.remove(parent.seq_id)
                for scheduler in self.scheduler:
                    # 释放该序列的KV-Cache分块表
                    scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                # 获取新的序列ID
                new_child_seq_id: int = next(self.seq_counter)
                # 复刻父序列并添加采样得到的token
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            # 在父序列上添加最后一个采样token
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            # 对每个子序列进行解码
            if sampling_params.detokenize and self.detokenizer:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, sampling_params)
            else:
                new_char_count = 0
            # 子序列判停
            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count,
                sampling_params,
                lora_req=seq_group.lora_request,
            )

        # Non-beam search case
        # 不使用beam-search(即生成多个输出序列)
        if not sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                # 父子序列为不同序列时
                if seq is not parent:
                    # 添加序列到序列组
                    seq_group.add(seq)
                    # 对于未完成的序列
                    if not seq.is_finished():
                        # 在每个PP-stage的调度器中复刻KV-Cache分块表
                        for scheduler in self.scheduler:
                            scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                # 如果扩展后的父序列已完成则移除其KV-Cache分块表
                # 注:单独一个for循环判断的原因是需要保证该父序列的子序列
                #    已经复刻完KV-Cache分块表之后才能释放,如果在一个循环中
                #    则可能先把父序列的KV-Cache分块表释放了
                if seq is parent and seq.is_finished():
                    for scheduler in self.scheduler:
                        scheduler.free_seq(seq)
            return

        # Beam search case
        # 使用 Beam-search
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
        unselected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
        beam_width = sampling_params.best_of
        length_penalty = sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        # 之前的和新生成的已完成序列列表
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                               reverse=True)
        # 遍历束宽范围内的已完成序列,选择新生成且已完成的序列到选中子序列列表
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        # 遍历束宽范围外的已完成序列
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # 新序列添加到未选中的子序列列表
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # 从序列分组中移除分数低的父序列
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=x[0].eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                sampling_params.early_stopping, sampling_params,
                best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        # 对选中的序列添加到序列组并复刻KV-Cache分块表
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    for scheduler in self.scheduler:
                        scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        # 释放选中但已完成的父序列KV-Cache分块表
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        # 对未选中的序列从序列组移除且释放KV-Cache分块表
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=current_worst_seq.eos_token_id)
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=best_running_seq.eos_token_id)
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=best_running_seq.eos_token_id))
        return current_worst_score >= highest_attainable_score
