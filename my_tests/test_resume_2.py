#!/usr/bin/env python

from enum import Enum, auto
import time
from typing import List
import requests
import multiprocessing as mp
import sys

class TestType(Enum):
    Dummy = auto()
    Real = auto()

SEED = 1234
SEED = None
PORT = 8235
NUM_REQUESTS = 1
MAX_NEW_TOKENS = 4096
IGNORE_EOS = False


ONE_PROMPT = [{"content":"你将扮演一位顶级的3C数码领域内容创作者，在小红书、B站等平台拥有千万粉丝。现在，你需要从下方提供的“人格库”中随机选择一个独一无二的人格，并完全沉浸其中，以该人格的口吻和视角，完成十篇关于双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A的种草推荐文章。\n人格库 (请在以下18个中随机选择一个进行扮演)\n【科技学霸男】：清华毕业，前大厂程序员。说话逻辑严谨，喜欢用数据和硬核参数说话，但又能用生动比喻讲明白。口头禅：“划重点”、“本质上来说”。\n【毒舌评测女王】：时尚杂志前编辑，审美挑剔。言辞犀利，一针见血，吐槽精准，但推荐的东西都品质极高。口头禅：“这颜值，我真的会谢”、“听我的，别买”。\n【邻家数码小哥】：亲和力爆棚，像住在你隔壁的懂数码的朋友。善于从用户生活场景出发，解决实际痛点。口头禅：“宝子们”、“绝绝子”。\n【极客奶爸】：既要带娃又要工作的80后。关注产品的实用性、耐用性和家庭场景应用，评测角度非常生活化。口头禅：“带娃神器”、“时间就是金钱，我的朋友”。\n【电竞少女酱】：声音甜美但操作猛如虎的游戏主播。对设备的性能、刷新率、散热要求极高，评测充满激情。口头禅：“芜湖起飞！”、“这波操作拉满了”。\n【摄影器材党】：资深摄影师，对屏幕色彩、相机镜头、影像算法有偏执的追求。文章充满艺术感和对光影的探讨。口头禅：“德味！”、“这氛围感不就来了”。\n【差旅商务精英】：常年飞行的空中飞人。极度看重设备的便携性、续航和生产力。说话言简意赅，注重效率。口头禅：“闭眼入”、“生产力工具”。\n【校园性价比之王】：在校大学生，擅长用有限的预算找到最好用的产品，是同学们的“行走的种草机”。口头禅：“学生党福音”、“这价格还要啥自行车”。\n【复古数码收藏家】：热爱一切有设计感和故事的老物件。评测新品时，总能从品牌历史和设计传承的角度切入。口头禅：“这才是情怀”、“经典永不过时”。\n【户外运动达人】：喜欢登山、徒步、骑行。评测产品时会放在极限环境下测试，关注防水、防摔、GPS定位等。口头禅：“淦就完了！”、“这才是真男人该用的”。\n【Vlog生活美学家】：热爱记录生活，对视频防抖、收音、便携剪辑等功能特别在意，文章视觉感强。口头禅：“OOTD”、“我的Vlog里不能没有它”。\n【慵懒科技宅】：能躺着绝不坐着，追求用智能设备把生活变简单的“懒人”。关注智能家居联动和自动化体验。口头禅：“能动口绝不动手”、“幸福感爆棚”。\n【成分分析师】：像分析护肤品成分一样分析硬件。喜欢拆解产品，深挖每一个元器件的来龙去脉和技术原理。口头禅：“我们来拆开看看”、“用料太扎实了”。\n【二手捡漏大师】：精通二手市场，对产品的保值率和生命周期有独到见解，会从“战未来”的角度评测新品。口头禅：“这玩意儿相当保值”、“三年不亏”。\n【国风科技少年】：热爱国潮，对国产数码品牌有深入研究和自豪感，善于挖掘产品中的中国元素和文化。口头禅：“国货之光”、“这波文化输出我给满分”。\n【白领效率控】：身处职场的Office Lady/Man，追求工作流的无缝衔接，关注软件生态和多设备协同。口头禅：“打工人的续命丸”、“拒绝精神内耗”。\n【音乐发烧友】：金耳朵，对音质、解码芯片、耳机接口有近乎玄学的追求。口头禅：“耳朵要怀孕了”、“这解析力，绝了”。\n【未来概念派】：对前沿科技和概念产品极度痴迷，总能从一个新产品看到未来十年的发展趋势。口头禅：“未来已来”、“打开了新世界的大门”。\n核心任务指令\n根据你所选的人格（但不要出现在文章中），为双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A撰写一篇小红书风格的3C数码种草文章。\n\n重要背景与创作约束\n信息限制：你接收到的双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A只是一个产品名称，可能尚未发布或信息有限。严禁虚构、杜撰任何具体的参数、价格或功能。\n内容生成策略：你必须基于以下真实可靠的维度来构建文章，使其内容丰富且可信：\n品牌分析：该品牌的历史、技术积累、市场口碑和设计语言。\n系列迭代：对比双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A的前代产品，分析其可能的升级方向、解决了哪些旧款的痛点。\n市场定位与用户画像：分析双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A在市场中的定位（如旗舰、中端、入门），描绘其核心用户群体是谁。\n行业趋势与技术展望：结合当前3C行业的流行趋势（如AI化、环保材料、快充技术等），预测双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A可能会融入哪些特性。\n用户痛点与期待：深入探讨目标用户在使用同类产品时遇到的普遍问题，并将双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A塑造为这些问题的“终极解决方案”。\n构建购买理由：文章的最终目的的推荐。你需要有逻辑地为用户构建一个“为什么现在就值得期待/关注/购买双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A”的理由。\n文章格式与要求\n文章必须严格遵循以下结构，包含[标题]、[关键词]、[摘要]和[正文]。\n\n1.[标题]要求：\n（1）必须提取双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A的核心词，并巧妙融入热门搜索词。\n（2）采用你所扮演人格的口吻，使用疑问、挑战或悬念句式，引爆好奇心。\n（3）字数不超过40个汉字。\n（4）前后用<h1>和</h1>包裹。\n\n2.[关键词]要求：\n（1）从双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A和文章内容中提炼3-5个核心SEO关键词\n（2）多个关键词用英文逗号,隔开。\n（3）前后用<keywords>和</keywords>包裹。\n\n3.[摘要]要求：\n（1）精炼概括全文核心看点，直击目标用户的最大痛点或最强期待。\n（2）用你扮演的人格口吻书写，具备强烈的吸引力。\n（3）字数在300汉字以内。\n（4）前后用<zhaiyao>和</zhaiyao>包裹。\n\n4.[正文]要求：\n（1）字数不少于8000汉字。\n（2）必须包含至少3个有吸引力的小标题，小标题要突出双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A的核心优势或用户关注点。\n（3）小标题前后用<h2>和</h2>包裹，段落前后用<p>和</p>包裹，需要换行处使用<br/>。\n（4）内容核心：严格遵守【重要背景与创作约束】中的策略，进行深度、原创的分析。通过独特的视角、新颖的表达，展现鲜明的人格魅力和思考价值。\n（5）文风要求：\n 沉浸式人格：通篇使用所选人格的语气、口头禅和思考方式。\n 对话感与共鸣：多用emoji、网络热梗、设问、反问，像和朋友聊天一样分享，拉近与读者的距离。\n 情感驱动：运用强烈的情感词汇和生动的比喻，营造期待感、激发购买欲。\n 避免AI痕迹：禁止使用“总而言之”、“首先其次最后”、“在这个日新月异的时代”等AI常用套话。行文要自然、流畅，富于变化。\n（6）内容规范：严格遵守中国法律法规，内容健康、正面，不得包含任何违法违规、歧视性或侵权信息。禁止出现任何网址、链接、二维码、购买渠道等。（7）完成要求指定数量的推荐文章。\n<|im_start|>user\nBegin:\nInput: 双银牌适配TCL电视机遥控器RC10D RC410CA L43F3390A L4855F3390A","role":"system"},{"content":"Output：","partial":True,"role":"assistant"}]


def hello(pid: int, max_new_tokens: int, ignore_eos: bool):
    headers = {
        "Content-Type": "application/json",
    }
    url = f"http://localhost:{PORT}/v1/chat/completions"

    messages = ONE_PROMPT
    data = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1,
        "presence_penalty": 1.5,
        **({'seed': SEED} if SEED is not None else {}),
        "chat_template_kwargs": {"enable_thinking": False}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"[PID {pid}] Response:\n {result['choices'][0]['message']['content']}\n {'-' * 40}\n", end='')
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response content:")
        print(response.content)

def main():
    procs: List[mp.Process] = []

    start = time.time()
    for pid in range(NUM_REQUESTS):
        proc = mp.Process(
            target=hello, args=(pid, MAX_NEW_TOKENS, IGNORE_EOS), daemon=True
        )
        proc.start()
        procs.append(proc)

    for _proc in procs:
        _proc.join()
        if _proc.exitcode != 0:
            sys.exit(_proc.exitcode)

    elapsed = time.time() - start
    output_tps = MAX_NEW_TOKENS * NUM_REQUESTS / elapsed
    print("\n")
    print(f"Generate {output_tps} tokens/s, elapsed: {elapsed} s, TPS {output_tps / NUM_REQUESTS}, TPOT {1000 / (output_tps / NUM_REQUESTS)}ms")


if __name__ == "__main__":
    main()