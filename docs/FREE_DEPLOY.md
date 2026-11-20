# Free deploy (no credit card): Hugging Face Spaces + Neon + Gemini free key

Costs nothing: Spaces (free CPU), Neon (free Postgres), Gemini free tier. Check each provider's current free limits.
Free Spaces go to sleep when unused and wake in about a minute when someone opens the link.

1. Load documents into Neon from your laptop (steps 10-12 of the main guide).
2. `python scripts/make_hf_space.py` creates the folder `hf_space_build`.
3. huggingface.co -> New Space -> name `knowledge-copilot`, SDK **Docker** (Blank), **Public**.
4. In the Space: Files -> Add file -> Upload files -> drag EVERYTHING inside `hf_space_build` (all files and folders) -> Commit.
5. Space -> Settings -> **Secrets**: `DATABASE_URL`, `GEMINI_API_KEY`, `JWT_SECRET` (any random 40+ characters).
   **Variables**: `PROVIDER=gemini`, `CHAT_MODEL=gemini-3.5-flash`, `DEMO_MODE=1`, `DEMO_ROLES=MEMBER,MANAGER,ADMIN`.
6. Wait for "Running". Your link: `https://<your-username>-knowledge-copilot.hf.space`.
