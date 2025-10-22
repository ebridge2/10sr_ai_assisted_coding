(intro:background)=
# Background concepts and vocabulary 

To navigate the challenges of AI-assisted coding effectively, researchers should be familiar with several key concepts that underpin these tools:

+ **Large Language Models (LLMs)** are neural networks trained on vast text corpora that generate text by predicting sequences of tokens, basic units of text processing that typically represent words, parts of words, or individual characters {cite}`vaswani2017attention`. For instance, the word "unhappily" might be tokenized as "un", "##happi", "##ly", where ## marks tokens that are not the start of a word.
+ **Context windows** define the maximum number of tokens an LLM can consider when generating responses. State-of-the-art models typically handle hundreds of thousands to millions of tokens, constraining how much code and documentation they can simultaneously process. When context limits are exceeded, models lose track of earlier information. Even when information is contained within the context window, attention to mid-document details can degrade ("lost in the middle"), especially for models with very large context windows; this phenomenon is known as **context rot**. For an example of context rot, see {cite}`chroma2024context`.
+ **In-context learning** allows models to adapt their behavior based on examples and instructions provided within the current conversation, without permanent changes to the underlying model. This enables direction of model behavior through strategic provision of examples and formatting of instructions.
+ **Prompting** encompasses techniques for structuring inputs to elicit desired outputs, including clear requirement specification, strategic provision of examples, and structured formatting. Effective prompting can dramatically improve code quality and relevance.
+ **Test-driven development** involves writing tests before implementation to specify expected behavior and validate correctness, a practice that becomes even more critical when AI generates the implementation code. Test driven development is detailed in {cite}`beck2003test`.

## Sharing context

AI coding tools range from conversational interfaces like ChatGPT to interactive assistants like GitHub Copilot to autonomous coding agents like Cursor. Each presents unique challenges for maintaining project context across sessions. Most AI systems are stateless, meaning they forget previous interactions, while others have limited understanding of broader project requirements. This creates two critical problems: context fragmentation, where important project details are lost between sessions, and iteration drift, where AI assistance gradually diverges from intended goals without proper oversight.

**Externally-managed context files** help address these limitations by providing persistent information across AI interactions. These include:

+ **Memory files** contain project-specific information like architectural decisions, software development standards and practices, and lessons learned that persist between interactions. They prevent repetition of past mistakes and ensure each new AI session starts with relevant context.
+ **Constitution files** establish non-negotiable principles governing AI behavior throughout development, such as security requirements or methodological constraints.

Together, these tools can help transform AI interactions into consistent, goal-directed collaboration by providing the persistent context and boundaries that AI systems lack natively.

---

## References

```{bibliography}
:filter: docname in docnames
```