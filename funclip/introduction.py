top_md_1 = ("""
    <div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
    FunClip: <a href='https://github.com/alibaba-damo-academy/FunClip'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
    🌟 Star Us: <a href='https://github.com/alibaba-damo-academy/FunClip/stargazers'><img src='https://img.shields.io/github/stars/alibaba-damo-academy/FunClip.svg?style=social'></a>
    </div>
    </div>

    Based on the [FunASR](https://github.com/alibaba-damo-academy/FunASR) toolkit developed by Alibaba Damo Academy with Paraformer model series, supporting speech recognition, voice activity detection, punctuation prediction, timestamp prediction, speaker diarization, and customized hotword setup.

    Accurate recognition, freely copy required segments, or set speaker identifiers, one-click clipping and subtitle addition.

    * Step1: Upload video or audio file (or use examples below), click the **<font color=\"#f7802b\">ASR</font>** button
    * Step2: Copy the desired text from recognition results to the upper right, or set speaker identifiers, adjust offsets and subtitle settings (optional)
    * Step3: Click **<font color=\"#f7802b\">Clip</font>** button or **<font color=\"#f7802b\">Clip and Add Subtitles</font>** button to get results

    🔥 FunClip now integrates LLM intelligent clipping functionality, try different LLM models~
    """)

top_md_3 = ("""Visiting the FunASR project and papers helps you understand the speech processing models used in FunClip:
    <div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        FunASR: <a href='https://github.com/alibaba-damo-academy/FunASR'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        FunASR Paper: <a href="https://arxiv.org/abs/2305.11013"><img src="https://img.shields.io/badge/Arxiv-2305.11013-orange"></a>
        🌟 Star FunASR: <a href='https://github.com/alibaba-damo-academy/FunASR/stargazers'><img src='https://img.shields.io/github/stars/alibaba-damo-academy/FunASR.svg?style=social'></a>
    </div>
    </div>
    """)

top_md_4 = ("""We provide three LLM invocation methods in the "LLM Intelligent Clipping" module:
            1. Select Alibaba Cloud Bailian platform to call Qwen series models via API. You need to prepare an API key from the Bailian platform, visit [Alibaba Cloud Bailian](https://bailian.console.aliyun.com/#/home);
            2. Select models starting with GPT to call OpenAI official API. You need to prepare your own API key and network environment;
            3. [gpt4free](https://github.com/xtekky/gpt4free?tab=readme-ov-file) project is also integrated into FunClip for free access to GPT models;

            Methods 1 and 2 require entering the corresponding API key in the interface.
            Method 3 may be very unstable with potentially long response times or failed results. You can try multiple times or prepare your own key using methods 1 or 2.

            Do not open multiple interfaces on the same port simultaneously, as it will cause file uploads to be very slow or freeze. Closing other interfaces will resolve this issue.
            """)
