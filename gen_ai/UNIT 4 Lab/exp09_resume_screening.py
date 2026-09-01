"""
Unit 4 - Experiment 9: AI-Based Resume Screening and Ranking
SIMATS Engineering | Name: G.Sai Teja | Reg No: 192472137

AIM:
    Develop an AI-based Resume Screening application that analyses candidate
    resumes and ranks them according to a given engineering job description.

METHOD:
    1. Semantic match : sentence-transformers/all-MiniLM-L6-v2 embeds the job
                        description and each resume; cosine similarity gives a
                        meaning-based match score (not just keyword overlap).
    2. Skill match    : the required skills listed in the JD are checked against
                        each resume text.
    3. Final score    : 70% semantic similarity + 30% skill coverage.

RUN:
    python exp09_resume_screening.py
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE = Path(__file__).parent
RESUME_DIR = BASE / "resumes"

JOB_TITLE = "Junior Machine Learning Engineer"
JOB_DESCRIPTION = """
We are hiring a Junior Machine Learning Engineer. The candidate must have a
B.E. or B.Tech in Computer Science, Information Technology or Artificial
Intelligence. Strong Python programming is essential, along with hands-on
experience in machine learning and deep learning using TensorFlow or PyTorch.
The role involves building and deploying models, data preprocessing with pandas
and NumPy, working with SQL databases and serving models through REST APIs
built with Flask or FastAPI. Familiarity with natural language processing,
computer vision, Git and cloud deployment on AWS is an added advantage.
"""

REQUIRED_SKILLS = [
    "python", "machine learning", "deep learning", "tensorflow", "pytorch",
    "pandas", "numpy", "sql", "flask", "fastapi", "nlp", "git", "aws",
]


def load_resumes():
    files = sorted(RESUME_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No resumes found in {RESUME_DIR}")
    return [(f.stem, f.read_text(encoding="utf-8")) for f in files]


def skill_coverage(resume_text):
    """Return (matched skills, missing skills) for one resume."""
    low = resume_text.lower()
    matched = [s for s in REQUIRED_SKILLS if s in low]
    missing = [s for s in REQUIRED_SKILLS if s not in low]
    return matched, missing


def main():
    print("Loading the pre-trained sentence embedding model...")
    model = SentenceTransformer(MODEL)

    resumes = load_resumes()
    print(f"Screening {len(resumes)} resumes for: {JOB_TITLE}\n")

    jd_vec = model.encode(JOB_DESCRIPTION, convert_to_tensor=True)

    results = []
    for name, text in resumes:
        resume_vec = model.encode(text, convert_to_tensor=True)
        semantic = float(util.cos_sim(jd_vec, resume_vec)[0][0])
        matched, missing = skill_coverage(text)
        skill_score = len(matched) / len(REQUIRED_SKILLS)
        final = 0.7 * semantic + 0.3 * skill_score
        results.append({
            "name": name, "semantic": semantic, "skill_score": skill_score,
            "matched": matched, "missing": missing, "final": final,
        })

    results.sort(key=lambda r: r["final"], reverse=True)

    print("=" * 70)
    print(f"{'Rank':<6}{'Candidate':<28}{'Semantic':>10}{'Skills':>10}{'Score':>10}")
    print("=" * 70)
    for rank, r in enumerate(results, start=1):
        skills = f"{len(r['matched'])}/{len(REQUIRED_SKILLS)}"
        print(f"{rank:<6}{r['name']:<28}{r['semantic']:>10.3f}{skills:>10}"
              f"{r['final'] * 100:>9.1f}%")
    print("=" * 70)

    print("\n--- Detailed report ---")
    for rank, r in enumerate(results, start=1):
        verdict = ("SHORTLIST" if r["final"] >= 0.55
                   else "MAYBE" if r["final"] >= 0.42 else "REJECT")
        print(f"\n{rank}. {r['name']}  [{verdict}]  score {r['final'] * 100:.1f}%")
        print(f"   Matched skills : {', '.join(r['matched']) or 'none'}")
        print(f"   Missing skills : {', '.join(r['missing']) or 'none'}")

    out = BASE / "outputs"
    out.mkdir(exist_ok=True)
    lines = [f"Job: {JOB_TITLE}", ""]
    lines += [f"{i}. {r['name']} - {r['final'] * 100:.1f}% "
              f"(matched: {', '.join(r['matched'])})"
              for i, r in enumerate(results, start=1)]
    (out / "exp09_ranking.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nRanking saved to: outputs/exp09_ranking.txt")


if __name__ == "__main__":
    main()
