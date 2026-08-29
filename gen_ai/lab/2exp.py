import google.generativeai as genai

# Enter your Gemini API key
API_KEY = input("Enter your Gemini API Key: ")

genai.configure(api_key=API_KEY)

# Create model
model = genai.GenerativeModel("gemini-2.5-flash")

# --------------------------------------------------
# 1. ZERO-SHOT PROMPT
# --------------------------------------------------

zero_shot_prompt = """
Write a 200-word blog on:
"Applications of Artificial Intelligence in Healthcare."

Explain the use of AI in:
1. Disease diagnosis
2. Medical imaging
3. Drug discovery
4. Patient monitoring
5. Personalized treatment
6. Healthcare administration

Use simple and informative language.
"""

# Generate Zero-shot output
zero_shot_output = model.generate_content(zero_shot_prompt)

print("\n========== ZERO-SHOT OUTPUT ==========")
print(zero_shot_output.text)


# --------------------------------------------------
# 2. ONE-SHOT PROMPT
# --------------------------------------------------

one_shot_prompt = """
Here is an example of the writing style:

Example:
Topic: Artificial Intelligence in Education

Blog:
Artificial Intelligence is transforming education by providing
personalized learning experiences. AI systems can analyze student
performance, recommend learning materials, automate assessments,
and help teachers identify students who need additional support.
As technology continues to develop, AI can make education more
effective and accessible.

Now write a 200-word blog on:

"Applications of Artificial Intelligence in Healthcare."

Discuss disease diagnosis, medical imaging, drug discovery,
patient monitoring, personalized treatment, and healthcare
administration.

Use the same simple and informative writing style as the example.
"""

# Generate One-shot output
one_shot_output = model.generate_content(one_shot_prompt)

print("\n========== ONE-SHOT OUTPUT ==========")
print(one_shot_output.text)


# --------------------------------------------------
# 3. FEW-SHOT PROMPT
# --------------------------------------------------

few_shot_prompt = """
Study the following examples and follow their style.

Example 1:
Topic: AI in Education

Blog:
Artificial Intelligence is changing education through
personalized learning, automated assessment, and intelligent
tutoring systems. It can analyze student performance and
recommend suitable learning resources.

Example 2:
Topic: AI in Banking

Blog:
Artificial Intelligence is improving banking by detecting fraud,
analyzing transactions, providing chatbots, and supporting
financial decisions. It helps banks provide faster and safer
services to customers.

Now write a 200-word blog on:

"Applications of Artificial Intelligence in Healthcare."

Include:
- Disease diagnosis
- Medical imaging
- Drug discovery
- Patient monitoring
- Personalized treatment
- Healthcare administration

Follow the writing style and structure of the examples.
Use simple and informative language.
"""

# Generate Few-shot output
few_shot_output = model.generate_content(few_shot_prompt)

print("\n========== FEW-SHOT OUTPUT ==========")
print(few_shot_output.text)


# --------------------------------------------------
# 4. COMPARISON
# --------------------------------------------------

print("\n========== COMPARISON ==========")

print("""
ZERO-SHOT:
No example is given.
The model depends only on the instructions.

ONE-SHOT:
One example is given.
The model uses that example to understand the
expected writing style and structure.

FEW-SHOT:
Multiple examples are given.
The model gets more guidance about the expected
style, structure, and content.
""")