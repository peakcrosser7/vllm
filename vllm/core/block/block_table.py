import math
from typing import List, Optional

from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.utils import Device, cdiv, chunk_list


class BlockTable:
    """A class to manage blocks for a specific sequence.

    The BlockTable maps a sequence of tokens to a list of blocks, where each
    block represents a contiguous memory allocation for a portion of the 
    sequence. The blocks are managed by a DeviceAwareBlockAllocator, which is
    responsible for allocating and freeing memory for the blocks.

    Args:
        block_size (int): The maximum number of tokens that can be stored in a
            single block.
        block_allocator (DeviceAwareBlockAllocator): The block allocator used to
            manage memory for the blocks.
        _blocks (Optional[List[Block]], optional): An optional list of existing
            blocks to initialize the BlockTable with. If not provided, an empty
            BlockTable is created.
        max_block_sliding_window (Optional[int], optional): The number of
            blocks to keep around for each sequance. If None, all blocks
            are kept (eg., when sliding window is not used).
            It should at least fit the sliding window size of the model.

    Attributes:
        _block_size (int): The maximum number of tokens that can be stored in a
            single block.
        _allocator (DeviceAwareBlockAllocator): The block allocator used to
            manage memory for the blocks.
        _blocks (Optional[List[Block]]): The list of blocks managed by this
            BlockTable.
        _num_full_slots (int): The number of tokens currently stored in the
            blocks.
    """

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        _blocks: Optional[List[Block]] = None,
        max_block_sliding_window: Optional[int] = None,
    ):
        self._block_size = block_size
        """KV-Cache分块大小"""
        self._allocator = block_allocator
        """分块分配器"""
        if _blocks is None:
            _blocks = []
        self._blocks: BlockList = BlockList(_blocks)
        """KV-Cache分块列表(对应一个序列中的连续token)"""

        self._max_block_sliding_window = max_block_sliding_window
        self._num_full_slots = self._get_num_token_ids()
        """已占用的token槽数(存储在分块表中的token数)"""


    @staticmethod
    def get_num_required_blocks(token_ids: List[int], block_size: int) -> int:
        """Calculates the minimum number of blocks required to store a given
        sequence of token IDs.

        This assumes worst-case scenario, where every block requires a new
        allocation (e.g. ignoring prefix caching).

        Args:
            token_ids (List[int]): The sequence of token IDs to be stored.
            block_size (int): The maximum number of tokens that can be stored in
                a single block.

        Returns:
            int: The minimum number of blocks required to store the given
                sequence of token IDs.
        """
        return cdiv(len(token_ids), block_size)

    def allocate(self,
                 token_ids: List[int],
                 device: Device = Device.GPU) -> None:
        """Allocates memory blocks for storing the given sequence of token IDs.

        This method allocates the required number of blocks to store the given
        sequence of token IDs.

        Args:
            token_ids (List[int]): The sequence of token IDs to be stored.
            device (Device, optional): The device on which the blocks should be
                allocated. Defaults to Device.GPU.
        """
        assert not self._is_allocated
        assert token_ids
        # 为token-IDs分配KV-Cache分块
        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_ids=token_ids,
                                                     device=device)
        # 用当前分块更替当前分块表
        self.update(blocks)
        self._num_full_slots = len(token_ids)

    def update(self, blocks: List[Block]) -> None:
        """Resets the table to the newly provided blocks 
        (with their corresponding block ids)
        """
        self._blocks.update(blocks)

    def append_token_ids(self,
                         token_ids: List[int],
                         num_lookahead_slots: int = 0,
                         num_computed_slots: Optional[int] = None) -> None:
        """Appends a sequence of token IDs to the existing blocks in the
        BlockTable.

        This method appends the given sequence of token IDs to the existing
        blocks in the BlockTable. If there is not enough space in the existing
        blocks, new blocks are allocated using the `ensure_num_empty_slots`
        method to accommodate the additional tokens.

        The token IDs are divided into chunks of size `block_size` (except for
        the first chunk, which may be smaller), and each chunk is appended to a
        separate block.

        Args:
            token_ids (List[int]): The sequence of token IDs to be appended.
            num_computed_slots (Optional[int]): The number of KV cache slots
                that are already filled (computed).
                When sliding window is enabled, this is used to compute how many
                blocks to drop at the front of the sequence.
                Without sliding window, None can be passed.
                Without chunked prefill, it should be the same as
                _num_full_slots.
        """
        assert self._is_allocated, "no blocks have been allocated"
        assert len(self._blocks) > 0

        # Drop blocks that are no longer needed due to sliding window
        if self._max_block_sliding_window is not None:
            null_block = self._allocator.allocate_or_get_null_block()
            assert num_computed_slots is not None
            # 滑动窗口的左边界
            end_block_idx = (num_computed_slots //
                             self._block_size) - self._max_block_sliding_window
            # 释放左边界以外的分块并设置为空分块
            for idx in range(0, end_block_idx):
                b: Block = self._blocks[idx]
                if b is not null_block:
                    self._allocator.free(b)
                    self._blocks[idx] = null_block

        # Ensure there are enough empty slots for the new tokens plus
        # lookahead slots
        # 确保分块表中有足够的空token槽,必要时分配KV-Cache分块
        self.ensure_num_empty_slots(num_empty_slots=len(token_ids) +
                                    num_lookahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = self._num_full_slots // self._block_size
        token_blocks: List[List[int]] = self._chunk_token_blocks_for_append(token_ids)

        for i, token_block in enumerate(token_blocks):
            self._blocks.append_token_ids(first_block_idx + i, token_block)

        # 更新已占用的token槽数
        # 注:num_lookahead_slots不占用token槽
        self._num_full_slots += len(token_ids)

    def ensure_num_empty_slots(self, num_empty_slots: int) -> None:
        """Ensures that the BlockTable has at least the specified number of
        empty slots available.

        This method checks if the BlockTable has enough empty slots (i.e.,
        available space) to accommodate the requested number of tokens. If not,
        it allocates additional blocks on the GPU to ensure that the required
        number of empty slots is available.

        Args:
            num_empty_slots (int): The minimum number of empty slots required.
        """
        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU
        assert self._is_allocated

        if self._num_empty_slots >= num_empty_slots:
            return

        # 可用token槽小于所需槽数则进行分配
        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)

        # 分配所需的KV-Cache(逻辑+物理)分块
        for _ in range(blocks_to_allocate):
            assert len(self._blocks) > 0
            self._blocks.append(
                self._allocator.allocate_mutable_block(
                    prev_block=self._blocks[-1], device=device))

    def fork(self) -> "BlockTable":
        """Creates a new BlockTable instance with a copy of the blocks from the
        current instance.

        This method creates a new BlockTable instance with the same block size,
        block allocator, and a copy of the blocks from the current instance. The
        new BlockTable has its own independent set of blocks, but shares the
        same underlying memory allocation with the original BlockTable.

        Returns:
            BlockTable: A new BlockTable instance with a copy of the blocks from
                the current instance.
        """
        assert self._is_allocated
        assert len(self._blocks) > 0
        # 指向相同物理分块的逻辑分块列表
        forked_blocks = self._allocator.fork(self._blocks[-1])
        return BlockTable(
            block_size=self._block_size,
            block_allocator=self._allocator,
            _blocks=forked_blocks,
            max_block_sliding_window=self._max_block_sliding_window,
        )

    def free(self) -> None:
        """Frees the memory occupied by the blocks in the BlockTable.

        This method iterates over all the blocks in the `_blocks` list and calls
        the `free` method of the `_allocator` object to release the memory
        occupied by each block. After freeing all the blocks, the `_blocks` list
        is set to `None`.
        """
        assert self._is_allocated
        # 遍历每个分块进行释放
        for block in self.blocks:
            self._allocator.free(block)
        self._blocks.reset()

    @property
    def physical_block_ids(self) -> List[int]:
        """Returns a list of physical block indices for the blocks in the
        BlockTable.

        This property returns a list of integers, where each integer represents
        the physical block index of a corresponding block in the `_blocks` list.
        The physical block index is a unique identifier for the memory location
        occupied by the block.

        Returns:
            List[int]: A list of physical block indices for the blocks in the
                BlockTable.
        """
        assert self._is_allocated
        return self._blocks.ids()

    def get_unseen_token_ids(self, sequence_token_ids: List[int]) -> List[int]:
        """Get the number of "unseen" tokens in the sequence.

        Unseen tokens are tokens in the sequence corresponding to this block
        table, but are not yet appended to this block table.

        Args:
            sequence_token_ids (List[int]): The list of token ids in the
                sequence.

        Returns:
            List[int]: The postfix of sequence_token_ids that has not yet been
                appended to the block table.
        """

        # Since the block table is append-only, the unseen token ids are the
        # ones after the appended ones.
        return sequence_token_ids[self.num_full_slots:]

    def _allocate_blocks_for_token_ids(self, prev_block: Optional[Block],
                                       token_ids: List[int],
                                       device: Device) -> List[Block]:
        """为token-IDs分配KV-Cache分块"""
        blocks: List[Block] = []

        block_token_ids = []    # 填充满的token分块
        tail_token_ids = []     # 尾部未填充满的token分块
        for cur_token_ids in chunk_list(token_ids, self._block_size):
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)

        if block_token_ids:
            # 对于已填满的token分块
            blocks.extend(
                # 创建不可更改的KV-Cache分块(即已经添加token-IDs的分块)
                self._allocator.allocate_immutable_blocks(
                    prev_block, block_token_ids=block_token_ids,
                    device=device))
            prev_block = blocks[-1]

        if tail_token_ids:
            # 对于尾部未填满的token分块
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            # 创建可更改的KV-Cache分块并添加token-IDs
            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block, device=device)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks

    def _get_all_token_ids(self) -> List[int]:
        # NOTE: This function is O(seq_len); use sparingly.
        token_ids: List[int] = []

        if not self._is_allocated:
            return token_ids

        for block in self.blocks:
            token_ids.extend(block.token_ids)

        return token_ids

    def _get_num_token_ids(self) -> int:
        """获取当前分块表中的token-ID总数"""
        res = 0
        for block in self.blocks:
            res += len(block.token_ids)

        return res

    @property
    def _is_allocated(self) -> bool:
        """是否分配了分块"""
        return len(self._blocks) > 0

    @property
    def blocks(self) -> List[Block]:
        """KV-Cache分块列表"""
        return self._blocks.list()

    @property
    def _num_empty_slots(self) -> int:
        """空余可用的token槽数"""
        assert self._is_allocated
        return len(self._blocks) * self._block_size - self._num_full_slots

    @property
    def num_full_slots(self) -> int:
        """Returns the total number of tokens currently stored in the
        BlockTable.

        Returns:
            int: The total number of tokens currently stored in the BlockTable.
        """
        return self._num_full_slots

    def get_num_blocks_touched_by_append_slots(
            self, token_ids: List[int], num_lookahead_slots: int) -> int:
        """Determine how many blocks will be "touched" by appending the token
        ids.

        This is required for the scheduler to determine whether a sequence can
        continue generation, or if it must be preempted.
        """
        # Math below is equivalent to:
        # all_token_ids = token_ids + [-1] * num_lookahead_slots
        # token_blocks = self._chunk_token_blocks_for_append(all_token_ids)
        # return len(token_blocks)

        # 需要添加的token总数
        num_token_ids = len(token_ids) + num_lookahead_slots
        # 第一个分块的大小,分块可能已存了一些token
        first_chunk_size = self._block_size - (self._num_full_slots %
                                               self._block_size)
        # 1表示的是第一个分块
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self._block_size))
        return num_token_blocks

    def _chunk_token_blocks_for_append(
            self, token_ids: List[int]) -> List[List[int]]:
        """Split the token ids into block-sized chunks so they can be easily
        appended to blocks. The first such "token block" may have less token ids
        than the block size, since the last allocated block may be partially
        full.

        If no token ids are provided, then no chunks are returned.
        """

        if not token_ids:
            return []

        # 第一个分块的可用token槽数
        first_chunk_size = self._block_size - (self._num_full_slots %
                                               self._block_size)
        token_blocks: List[List[int]] = [token_ids[:first_chunk_size]]
        token_blocks.extend(
            # 按照分块大小对token_ids进行切块
            chunk_list(token_ids[first_chunk_size:], self._block_size))
        return token_blocks
