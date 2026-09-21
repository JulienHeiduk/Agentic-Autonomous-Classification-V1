# Reference: 1st Place, "Distributed Intelligence", NVIDIA Inference Hub (Playground S6E8)

Source: https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe
Text pasted by the project owner on 2026-09-06 (the page is client-rendered and not reachable
through the Kaggle API). Reproduced verbatim for reference; all credit to the author.

---

Thank you Kaggle for a fun Playground competition. This month, I used distributed intelligence to win 1st place. Phase 1 used a single autonomous agent, phase 2 added 2 more agents and a shared knowledge database. Then phase 3 and phase 4 added a swarm of new agents. Using NVIDIA Inference Hub, we can pick from roughly 150 different LLM models in our agent harness!

## Phase 1 - Autonomous Agent

I joined this competition 1.5 weeks ago and began using a single Codex GPT 5.6 Sol autorun agent (similar to what I did in Wellbore competition here). I instructed the agent to do everything. It read the competition description, downloaded the data and began building a big diverse ensemble (i.e. a solid strategy in playground comp). Every time it added another 10-50 group of models, it would compute new ensemble CV score and submit to Kaggle LB by itself. After 4 days, it had built a 380 model ensemble and reached 10th place (or better, i forget) on public LB by itself!

## Phase 2 - GPT 5.6 Sol battles Fable 5

I worked with Fable 5 in Kaggle's NeuroGolf competition here and was mind blown with its abilities. In phase 2, we onboarded Fable 5 and another GPT 5.6 Sol. We told them to perform EDA and read our previous work. We then challenged Codex GPT 5.6 Sol to compete against Claude Code Fable 5 to see who can make the best single NN model and who could make the best single XGB model. For the next week they battled it out. Mostly they would work alone. When one was lagging, we would share tips from the leading agent to the trailing agent and soon the trailing agent would take the lead! We achieved single model RealMLP CV 0.97070 LB 0.97174 and single model XGB CV 0.97020 LB 0.97030, wow amazing!

## Single Model wins 1st Place!

To my amazement, GPT 5.6 Sol collaborating (i.e. competing and occasionally sharing) with Fable 5 created a single model that by itself (without ensemble) wins 1st place in Kaggle's August Playground competition. A single model has not won Kaggle's Playground competition in 18 months! (The last was February 2025 here). This is a huge achievement and confirms that frontier models are Grandmasters at data science modeling! Furthermore, we can add every single model they built to our large ensemble to supercharge its performance!

## Phase 3 - ChatGPT Pro - Swarm Intelligence!

To help GPT 5.6 Sol and Fable 5 build even better single models, we asked Fable to build a tar.gz packet and write a prompt that we (humans) can (manually) upload to ChatGPT Pro to solicit its help. (This worked well in NeuroGolf competition here). Since ChatGPT Pro can answer multiple questions in parallel, we can submit multiple packets and multiple prompts in parallel. ChatGPT Pro works for 2 hours (on each packet/prompt) and then creates a ZIP file for us to download with its discoveries. When GPT 5.6 Sol and Fable 5 incorporated ChatGPT Pro insights into their work, they improved their best single model CV and LB scores!

I was amazed to witness that ChatGPT Pro discovered new tabular data feature engineering ideas that I have never seen before in my 8 years of competing in tabular data competitions on Kaggle. These are techniques that apply to this competition and every future tabular data science competition. Absolutely amazing!

## Phase 4 - NVIDIA Inference Hub - Swarm Intelligence!

To help us improve our ensemble's CV and LB score even more, we asked Fable 5 to interview new LLMs from NVIDIA Inference Hub's roughly 150 potential LLM and then enlist them to help us. (Using a variety of LLM worked well in MAP competition here). Fable 5 assessed the skills of

- Nemotron 3 Ultra
- DeepSeek V4 Pro
- Kimi K3
- Gemini 3.7 Flash
- Opus 5
- Qwen 3.8 27B

After assessing each LLM's data science abilities, Fable 5 assigned each LLM multiple jobs to improve our ensemble's CV score and LB score.

## Conclusion - Mind Blown!

In conclusion, I am completely mind blown at the ability of today's frontier LLMs. I have been a top Kaggle competitions Grandmaster for 6+ years now and I can say that the work I witnessed these agents perform in this competition exceeds what I believe is humanly possible. We are certainly in the new era of agentic data science.
