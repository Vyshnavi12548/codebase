# codebase
This project is for CredTech Hackathon organised by  Indian Institute of Technology (IIT), Kanpur
CredTech: Your Explainable Credit Score Co-pilot
What We Built 🚀
Imagine a world where a company's credit score isn't a mysterious number from a slow, traditional agency. We built a platform that gives you an updated, transparent credit score in seconds. It's like a real-time health tracker for a company, telling you not just what the score is, but why it is that way.

Our web app makes this all super simple. Just type in a company name, and our "credit co-pilot" gets to work, pulling in all the latest info to give you a clear, easy-to-understand breakdown.

How it Works (in a Nutshell) 💡
It Gathers All the Clues: Our system is a digital detective. It doesn't just look at old financial reports. It simultaneously checks:

Company Finances: The most recent revenue, debt, and market data.

News & Buzz: What's being said in the news, to figure out public sentiment.

Tech Signals: Our unique twist! We check a company's open-source activity on platforms like GitHub. We believe a healthy, engaged code community is a sign of a strong, innovative company.

It Connects the Dots: All these clues are fed into a smart prediction engine. Think of it as a seasoned analyst who has learned to weigh all these different factors. It crunches the numbers and delivers a credit score based on what it's learned from real data.

It Gives a Simple Explanation: No black boxes here! For every score, the app shows you exactly what clues had the biggest impact, both good and bad. You get a chart that breaks down the score's key drivers, so you can see the reasoning at a glance.

The Brains Behind the Operation 🧠
Our tech stack was chosen for speed and clarity.

Streamlit: We used this for the app's dashboard because it let us build a beautiful, interactive web app with just one Python file. This meant we could focus on the core problem without getting bogged down in complex web development.

Machine Learning Model: Instead of using simple rules, we trained a machine learning model called RandomForestRegressor. This model is much better at finding complex patterns and giving a more accurate, nuanced score.

SHAP Explainability: This is our secret weapon for transparency. SHAP is a powerful tool that tells us exactly how our model made its decision. It’s what powers the "why" in our explanations.

Getting it Running on Your Computer 🧑‍💻
Don't worry, it's easy!

Get Your Tools Ready: Make sure you have Python (3.8 or newer) and Docker installed. You'll also need a couple of free API keys for News API and GitHub.

Get the Code: Grab all the project files from our repository.

Install the Libraries: Open your terminal and run the command to install all the Python libraries our project needs.

Train the Brain: We've included a simple script to get the machine learning model ready. Just run one command to train it.

Launch the App: Use the streamlit command to start the app in your browser, and you're good to go!

We also included a Dockerfile so you can easily package the entire application. It's like putting your app into a portable box, ready to be deployed anywhere.

What's Next? 🚀
This is just the beginning. Our vision for this platform includes:

Automated Updates: The ability to automatically refresh data and retrain the model continuously.

Advanced Signals: Integrating more unique data sources, like satellite imagery or trade flow data.

Smarter Explanations: Using more sophisticated NLP to provide even richer, more human-like reasoning for each score.

