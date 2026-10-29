# Model provider marks

These locally bundled marks identify model providers, not endorsements or partnerships.

- `eleutherai.png`: white-on-black mark from [EleutherAI's website repository](https://github.com/EleutherAI/eleutherai.github.io/blob/master/images/eai_logo.png), downscaled from 1024 to 128 pixels for its 24 to 32 pixel display size; otherwise unmodified.
- `openai.png`: unmodified avatar from [OpenAI's verified Hugging Face organization](https://huggingface.co/openai), retrieved from its public organization profile on 2026-09-06. [OpenAI brand guidelines](https://openai.com/brand/).

`ModelProviderLogo.svelte` uses OpenAI's alpha channel and EleutherAI's luminance for monochrome display. It retains the existing Gemma and Qwen vector marks. All marks are decorative beside a readable model name; no remote image request is needed at runtime.
