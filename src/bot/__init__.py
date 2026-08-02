"""Tool-calling analytics assistant."""

# Intentionally empty of eager imports. Pulling agent/tools here would create a
# circular import once analytics code (via the Redis cache) and the bot package
# need each other. Import `answer_question` / `stream_answer` from
# `src.bot.agent` directly.
