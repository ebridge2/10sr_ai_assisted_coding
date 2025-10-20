# Ten Simple Rules for AI-assisted Coding in Open Science

While AI coding tools have demonstrated potential to accelerate development, their use in scientific computing raises critical questions about code quality, domain appropriateness, and scientific validity. This paper provides ten practical rules for AI-assisted coding that balance leveraging capabilities of AI with maintaining scientific and methodological rigor. We address how AI can be leveraged strategically throughout the development cycle with four key themes: problem preparation and understanding, managing context and interaction, testing and validation, and code quality assurance and iterative improvement. Our focal principles emphasize maintaining human agency in coding decisions, establishing robust validation procedures, and preserving the domain expertise essential for methodologically sound research. Together, we help researchers harness AI's transformative potential for faster development while ensuring their code meets the standards of reliability, reproducibility, and scientific appropriateness that research integrity demands.

## Usage

This jupyter book provides a complementary extended version of our ongoing ten simple rules paper. It includes expanded discussions of the individual rules; particularly, focusing on both positive (and negative) examples of the rules look like in practice. While our examples focus on scientific problems in neuroimaging and cognitive sciences (the backgrounds of many of the co-authors on this work), the principles espoused in these rules extend directly to other domains of science where reproducibility and future implementability of produced code derivatives is an important focus. 

If you'd like to develop and build the existing draft of our ten simple rules paper, you should:

+ Clone the repository locally,
+ Optionally, set up a local virtual environment (e.g., with `virtualenv`) for the book,
+ Install the dependencies: `pip install jupyter-book sphinx-proof sphinx-togglebutton`,
+ Run `jupyter-book build paper/`.

A fully rendered version of the book will be built in `paper/_build/html/index.html`.

## Contributing

We welcome and recognize contributions to our ongoing project. You can see the list of current contributors to the current version of the webpage [contributors tab](https://github.com/ebridge2/10sr_ai_assisted_coding/graphs/contributors), and co-authors of our paper are provided in the [coverpage](https://github.com/ebridge2/10sr_ai_assisted_coding/blob/main/paper/coverpage.md). If you have simple formatting recommendations, we would love to have your proposed updates. If you have impactful content you would like to share, let us know, and we can discuss potential inclusion in our ongoing paper. 
