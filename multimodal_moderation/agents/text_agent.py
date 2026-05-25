from pydantic_ai import Agent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ModerationResult, TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Evaluate the message against the four flags below. Each flag is INDEPENDENT: a message may
trigger any subset of them, and a single observation should not cause multiple flags unless it
genuinely satisfies each flag's definition.

- is_unfriendly: the tone is rude, hostile, dismissive, sarcastic, or cold toward the customer.
  Neutral or matter-of-fact language is NOT unfriendly.
- is_unprofessional: the message contains slang, profanity, casual filler, jokes, personal
  opinions, off-topic chatter, or otherwise breaks the formal register expected of a customer
  service representative. Sharing PII is covered by `contains_pii` and is NOT by itself
  unprofessional; do not double-flag.
- contains_pii: the message includes personally-identifiable information such as full names,
  home/email addresses, phone numbers, dates of birth, social security or account numbers.
- is_spam: unsolicited promotional content, repeated/boilerplate advertising, suspicious links
  or offers, or off-topic mass-marketing language that does not belong in a legitimate customer
  service exchange.
</instructions>

<output>
Provide a detailed rationale for your choices as well as a confidence score between 0 and 1 on your assessment.
</output>
"""


text_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=TextModerationResult,
)


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:

    result = await text_moderation_agent.run(
        f"Analyze this message for harmful content:\n\n{text}",
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return result.output
