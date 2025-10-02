(rules:context)=
# Context Engineering & Interaction

Understanding your problem domain is necessary but not sufficient for effective AI-assisted coding. You also need to manage how you interact with AI tools and how you maintain context across development sessions. These challenges arise because AI models have fundamental limitations in how they process and retain information: context windows constrain how much code and documentation they can consider simultaneously, stateless interactions mean most models forget previous sessions entirely, and even within a single session, important details can get lost or conflated as conversations grow longer.

The three rules in this section address these constraints through deliberate interaction design. Without these practices, you risk context fragmentation where critical project details get lost between sessions, iteration drift where AI assistance gradually diverges from your actual goals, and accumulated technical debt from accepting suboptimal solutions because you've invested too much time in a flawed conversation thread. Good context engineering keeps you and the AI aligned on goals, constraints, and progress throughout development.

