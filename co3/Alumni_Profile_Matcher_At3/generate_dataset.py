"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic dataset of 220 alumni bios for the
"Alumni Profile Matcher" benchmark.

Design notes (important for the special-case query):
- Every alumnus has a `company` and a `role`, and the `bio` text is written
  the way a real alumnus would write it -- i.e. it usually names the COMPANY
  and the ROLE, and only SOMETIMES explicitly uses a domain buzzword like
  "fintech", "edtech", "healthtech".
- A subset of alumni work at well-known payments/lending companies
  (Razorpay, Stripe, PayU, Paytm, CRED, Visa) but their bios talk about
  "backend systems", "distributed ledgers", "risk models", "checkout flow"
  etc. WITHOUT ever using the word "fintech". These are the special-case
  records: a student query for "fintech" should only be able to find them
  through semantic/company-domain association, not keyword overlap.
- Source: this is a generated (synthetic) dataset created for this
  assignment -- no real alumni data is used. Company names are real-world
  well-known firms used only to give the embedding model realistic
  domain signal (a common, accepted practice for synthetic benchmark data).
"""

import json
import random

random.seed(42)

FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Krishna","Ishaan","Rohan",
    "Ananya","Diya","Isha","Kavya","Meera","Priya","Riya","Saanvi","Tara","Zara",
    "Karthik","Naveen","Suresh","Vikram","Rahul","Sanjay","Arun","Deepak","Manoj","Gopal",
    "Divya","Lakshmi","Nisha","Pooja","Radha","Sneha","Swathi","Vidya","Anjali","Bhavana",
    "Nikhil","Siddharth","Varun","Yash","Akash","Dhruv","Kabir","Aryan","Om","Raghav",
]
LAST_NAMES = [
    "Sharma","Iyer","Reddy","Nair","Menon","Rao","Krishnan","Pillai","Gupta","Verma",
    "Subramanian","Chandran","Balan","Raghavan","Venkatesh","Mahesh","Suresh","Das","Sen","Bose",
]

def name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

BATCH_YEARS = list(range(2005, 2023))
DEGREES = ["B.Tech CSE", "B.Tech ECE", "B.Tech Mechanical", "B.Tech EEE", "B.Tech IT",
           "MBA", "M.Tech CSE", "M.Sc Data Science"]

# ---------------------------------------------------------------------
# Domain definitions. Each domain has:
#   companies: list of real-world company names typical of that domain
#   roles: typical job titles
#   templates: bio sentence templates. `explicit` templates use the
#              buzzword (e.g. "fintech"); `implicit` templates never do.
# ---------------------------------------------------------------------
DOMAINS = {
    "fintech": {
        "companies": ["Razorpay", "Stripe", "PayU", "Paytm", "CRED", "Visa", "PhonePe", "BillDesk"],
        "roles": ["Backend Engineer", "Risk Analyst", "Product Manager", "Data Scientist",
                  "Site Reliability Engineer", "Engineering Manager"],
        "explicit": [
            "{n} works as a {r} at {c}, building fintech products used by millions of merchants.",
            "{n} is a {r} in the fintech space at {c}, focused on digital payments infrastructure.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, designing high-throughput checkout and settlement systems for online merchants.",
            "{n} works at {c} as a {r}, building fraud-detection models for card and UPI transactions.",
            "{n} leads a team at {c} that manages the distributed ledger and reconciliation pipeline for merchant payouts.",
            "{n} is a {r} at {c}, working on real-time risk scoring for lending and credit-line products.",
            "{n} builds the payment-gateway APIs and webhook infrastructure at {c} as a {r}.",
            "{n} works on wallet top-ups, KYC verification, and settlement banking rails at {c}.",
        ],
    },
    "healthtech": {
        "companies": ["Practo", "Apollo 24/7", "Pfizer", "Philips Healthcare", "1mg", "Cure.fit"],
        "roles": ["Product Manager", "ML Engineer", "Clinical Data Analyst", "Software Engineer"],
        "explicit": [
            "{n} is a {r} at {c}, building healthtech solutions for patient care.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, building appointment-scheduling and teleconsultation platforms for hospitals.",
            "{n} is a {r} at {c}, analysing electronic health records to predict patient readmission risk.",
            "{n} designs diagnostic imaging software at {c} as a {r}.",
        ],
    },
    "edtech": {
        "companies": ["BYJU'S", "Unacademy", "Coursera", "upGrad", "Vedantu", "Khan Academy"],
        "roles": ["Software Engineer", "Content Product Manager", "Data Analyst", "ML Engineer"],
        "explicit": [
            "{n} works in edtech at {c} as a {r}, building online learning products.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, building adaptive quiz engines and personalised video recommendation systems.",
            "{n} works at {c} as a {r}, improving classroom livestreaming and doubt-resolution tools for students.",
        ],
    },
    "core_software": {
        "companies": ["Google", "Microsoft", "Amazon", "Adobe", "VMware", "Atlassian", "Oracle"],
        "roles": ["Software Development Engineer", "Senior SDE", "Staff Engineer", "Engineering Manager", "Tech Lead"],
        "explicit": [
            "{n} is a {r} at {c}, working on core software infrastructure and cloud platforms.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, building distributed storage systems that scale to billions of requests.",
            "{n} is a {r} at {c}, optimising compiler toolchains and low-latency runtime systems.",
            "{n} leads a team at {c} responsible for the internal developer-tooling platform.",
            "{n} works on search ranking and indexing pipelines at {c} as a {r}.",
        ],
    },
    "data_ai": {
        "companies": ["NVIDIA", "OpenAI", "DeepMind", "Meta AI", "Anthropic", "Hugging Face", "Databricks"],
        "roles": ["Machine Learning Engineer", "Research Scientist", "Data Scientist", "MLOps Engineer", "AI Researcher"],
        "explicit": [
            "{n} is a {r} at {c}, working on artificial intelligence and machine learning research.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, training large language models and evaluating their reasoning abilities.",
            "{n} is a {r} at {c}, building recommendation systems using deep learning and graph neural networks.",
            "{n} works on GPU kernel optimisation and distributed training infrastructure at {c}.",
            "{n} builds computer-vision pipelines for autonomous perception systems at {c} as a {r}.",
        ],
    },
    "consulting": {
        "companies": ["McKinsey & Company", "Boston Consulting Group", "Bain & Company", "Deloitte", "EY", "Accenture"],
        "roles": ["Management Consultant", "Strategy Analyst", "Associate Consultant", "Engagement Manager"],
        "explicit": [
            "{n} is a {r} at {c}, advising Fortune 500 clients on business strategy.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, helping manufacturing clients redesign their supply-chain operations.",
            "{n} is a {r} at {c}, running digital-transformation programmes for retail and telecom clients.",
            "{n} advises private-equity clients on due diligence and post-merger integration at {c}.",
        ],
    },
    "core_engineering": {
        "companies": ["L&T", "Tata Motors", "Mahindra & Mahindra", "ISRO", "BHEL", "Siemens Energy", "GE Aerospace"],
        "roles": ["Design Engineer", "Project Engineer", "R&D Engineer", "Systems Engineer", "Manufacturing Engineer"],
        "explicit": [
            "{n} is a {r} at {c}, working in core mechanical and industrial engineering.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, designing propulsion systems and running structural stress simulations.",
            "{n} is a {r} at {c}, leading assembly-line automation and quality-control processes for vehicle manufacturing.",
            "{n} designs turbine components and thermal systems at {c} as a {r}.",
        ],
    },
    "ecommerce_logistics": {
        "companies": ["Flipkart", "Amazon Retail", "Myntra", "Delhivery", "Blinkit", "Zepto", "Meesho"],
        "roles": ["Product Manager", "Supply Chain Analyst", "Software Engineer", "Operations Manager"],
        "explicit": [
            "{n} works in e-commerce at {c} as a {r}, running online retail operations.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, optimising last-mile delivery routes and warehouse fulfilment systems.",
            "{n} works at {c} as a {r}, building demand-forecasting models for inventory planning.",
            "{n} leads the dark-store and quick-commerce logistics network at {c} as a {r}.",
        ],
    },
    "cybersecurity": {
        "companies": ["Palo Alto Networks", "CrowdStrike", "Cisco", "Fortinet", "Zscaler", "IBM Security"],
        "roles": ["Security Engineer", "Threat Analyst", "Penetration Tester", "Security Architect"],
        "explicit": [
            "{n} works in cybersecurity at {c} as a {r}, protecting enterprise networks from threats.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, hunting for zero-day vulnerabilities in cloud infrastructure.",
            "{n} works at {c} as a {r}, building intrusion-detection and SOC automation tooling.",
        ],
    },
    "academia_research": {
        "companies": ["IIT Madras", "MIT", "Stanford University", "IISc Bangalore", "Carnegie Mellon University"],
        "roles": ["PhD Researcher", "Postdoctoral Fellow", "Assistant Professor", "Research Associate"],
        "explicit": [
            "{n} is an {r} at {c}, researching applied mathematics and academic theory.",
        ],
        "implicit": [
            "{n} works at {c} as a {r}, publishing papers on control theory and robotics.",
            "{n} teaches and researches signal processing and communication systems at {c}.",
        ],
    },
    "marketing_media": {
        "companies": ["Ogilvy", "WPP", "Dentsu", "Star India", "Netflix", "Spotify"],
        "roles": ["Brand Manager", "Growth Marketer", "Content Strategist", "Product Marketing Manager"],
        "explicit": [
            "{n} works in digital marketing at {c} as a {r}, running brand campaigns.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, running performance-marketing campaigns and analysing conversion funnels.",
            "{n} works at {c} as a {r}, producing original content strategy for streaming audiences.",
        ],
    },
    "energy_climate": {
        "companies": ["Tata Power", "ReNew Power", "Shell", "ONGC", "Adani Green Energy"],
        "roles": ["Energy Analyst", "Project Engineer", "Sustainability Consultant", "Grid Engineer"],
        "explicit": [
            "{n} works in renewable energy at {c} as a {r}, driving the clean-energy transition.",
        ],
        "implicit": [
            "{n} is a {r} at {c}, designing solar-farm layouts and grid-integration studies.",
            "{n} works at {c} as a {r}, modelling carbon-emission reduction pathways for industrial clients.",
        ],
    },
}

INTERESTS_POOL = {
    "fintech": ["digital payments", "UPI systems", "fraud detection", "lending platforms", "financial inclusion"],
    "healthtech": ["digital health", "telemedicine", "medical imaging", "patient data analytics"],
    "edtech": ["online learning", "personalised education", "ed-tech platforms"],
    "core_software": ["distributed systems", "cloud computing", "backend engineering", "developer tools"],
    "data_ai": ["machine learning", "deep learning", "NLP", "computer vision", "MLOps"],
    "consulting": ["business strategy", "management consulting", "digital transformation"],
    "core_engineering": ["mechanical design", "manufacturing", "robotics", "automotive engineering"],
    "ecommerce_logistics": ["supply chain", "logistics tech", "online retail", "quick commerce"],
    "cybersecurity": ["network security", "threat intelligence", "penetration testing"],
    "academia_research": ["applied research", "academia", "control systems", "signal processing"],
    "marketing_media": ["digital marketing", "brand strategy", "content strategy"],
    "energy_climate": ["renewable energy", "sustainability", "clean tech"],
}

def make_record(idx, domain_key, explicit_ratio=0.35):
    d = DOMAINS[domain_key]
    n = name()
    c = random.choice(d["companies"])
    r = random.choice(d["roles"])
    use_explicit = bool(d["explicit"]) and random.random() < explicit_ratio
    template = random.choice(d["explicit"] if use_explicit else d["implicit"])
    bio_core = template.format(n=n, r=r, c=c)
    year = random.choice(BATCH_YEARS)
    degree = random.choice(DEGREES)
    interests = random.sample(INTERESTS_POOL[domain_key], k=min(2, len(INTERESTS_POOL[domain_key])))
    tail = f" A {degree} graduate (Class of {year}), {n.split()[0]} enjoys mentoring students interested in {', '.join(interests)}."
    bio = bio_core + tail
    return {
        "id": f"alum_{idx:04d}",
        "name": n,
        "domain": domain_key,
        "company": c,
        "role": r,
        "batch_year": year,
        "degree": degree,
        "explicit_domain_wording": use_explicit,
        "bio": bio,
    }

def build_dataset(n_total=220):
    records = []
    domain_keys = list(DOMAINS.keys())
    idx = 1
    # ensure every domain gets a healthy share
    per_domain = n_total // len(domain_keys)
    for dk in domain_keys:
        for _ in range(per_domain):
            records.append(make_record(idx, dk))
            idx += 1
    # top up to reach n_total
    while len(records) < n_total:
        records.append(make_record(idx, random.choice(domain_keys)))
        idx += 1
    random.shuffle(records)
    # renumber ids after shuffle for a clean sequential id in file order
    for i, rec in enumerate(records, start=1):
        rec["id"] = f"alum_{i:04d}"
    return records

if __name__ == "__main__":
    data = build_dataset(220)
    with open("alumni_bios.json", "w") as f:
        json.dump(data, f, indent=2)

    n_fintech_implicit = sum(
        1 for r in data if r["domain"] == "fintech" and not r["explicit_domain_wording"]
    )
    n_fintech_total = sum(1 for r in data if r["domain"] == "fintech")
    print(f"Total records: {len(data)}")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Fintech records: {n_fintech_total} (implicit/no-buzzword: {n_fintech_implicit})")
    from collections import Counter
    print("Per-domain counts:", Counter(r["domain"] for r in data))
