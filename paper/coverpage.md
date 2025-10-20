# Ten Simple Rules for AI-assisted Coding in Science

Eric W. Bridgeford, Iain Campbell, Zijiao Chen, Harrison Ritz, Joachim Vandekerckhove, Russell Poldrack

## Abstract

While AI coding tools have demonstrated potential to accelerate development, their use in scientific computing raises critical questions about code quality, domain appropriateness, and scientific validity. These challenges are compounded by AI systems' inherent limitations: constrained context windows, stateless interactions that lose track of previous exchanges, and pattern-matching behavior that cannot substitute for domain reasoning. This paper provides ten practical rules for AI-assisted coding that balance leveraging capabilities of AI with maintaining scientific and methodological rigor. We address how AI can be leveraged strategically throughout the development cycle with four key themes: problem preparation and understanding, managing context and interaction, testing and validation, and code quality assurance and iterative improvement. These focal principles emphasize maintaining human agency in coding decisions, establishing robust validation procedures, and preserving the domain expertise essential for methodologically sound research. Together, we help researchers harness AI's transformative potential for faster development while ensuring their code meets the standards of reliability, reproducibility, and scientific appropriateness that research integrity demands.

## How to use this jupyter book

This Jupyter Book complements our paper "Ten Simple Rules for AI-Assisted Coding in Science" by providing concrete, realistic examples of each rule in action. While the paper articulates principles and rationale, this resource shows you *how to apply them in practice*.

The difference between understanding a principle and applying it effectively often lies in seeing concrete examples. In our paper, space constraints limit how thoroughly we can demonstrate each rule through realistic scenarios. Here, we provide extended examples that show:

- **What good looks like**: Positive examples demonstrating effective application of each rule
- **What to avoid**: Flawed examples showing common pitfalls and their consequences  
- **The distinction between them**: Explicit explanations of what separates effective from ineffective approaches

For each rule, you'll find:

- **The rule itself**: Restated from the paper for easy reference.
- **What separates positive from flawed examples**: A brief explanation of the key distinctions between good, and flawed, examples relating to the rule.
- **Multiple examples**: Realistic scenarios showing both effective and ineffective applications, presented as realistic conversational interactions with AI tools. An important note: when we the authors interact with AI language models, *flawed interactions happen more often than successful ones*. This is normal. Don't be discouraged; learn from what went wrong, refine your approach, and try again. The examples here are sometimes simplified to clearly highlight what makes an interaction effective or flawed; they're not meant to show complete problem-solving workflows from start to finish.



## Usage

This jupyter book provides a complementary extended version of our ongoing ten simple rules paper. It includes expanded discussions of the individual rules; particularly, focusing on both positive (and negative) examples of the rules look like in practice. While our examples focus on scientific problems in neuroimaging and cognitive sciences (the backgrounds of many of the co-authors on this work), the principles espoused in these rules extend directly to other domains of science where reproducibility and future implementability of produced code derivatives is an important focus. 

If you'd like to develop and build the existing draft of our ten simple rules paper, you should:

+ Clone the repository locally,
+ Optionally, set up a local virtual environment (e.g., with `virtualenv`) for the book,
+ Install the dependencies: `pip install jupyter-book sphinx-proof sphinx-togglebutton`,
+ Run `jupyter-book build paper/`.

A fully rendered version of the book will be built in `paper/_build/html/index.html`.
