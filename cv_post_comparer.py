import os
import json
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from datetime import datetime
from prompts.prompt_template import RESUME_TEXT_TO_JSON_PROMPT, MATHCH_JOB_AND_RESUME_PROMPT
from schema import MatchResult
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")





job_description = """
    The Position: 
    About Roche Digital Technology (RDT)
    Roche Digital Technology (RDT) is where innovation meets purpose. As a global team at the heart of Roche, we are a community of business-minded technologists committed to help shape tomorrow’s digital future of healthcare. Our mission is to power Roche through cutting-edge digital technologies, harnessing the potential of artificial intelligence, data, and scalable tech innovations. Driven by purpose and passion, we’re building a future where digital is a core strength across all of Roche, enabling smarter ways of working, unlocking human potential, and driving breakthroughs that truly matter for millions of patients around the world.

    Role Summary: 
    The responsible and ethical use of Artificial Intelligence is paramount to achieving this goal. The Head of AI Risks & Ethics is a senior leadership role, reporting directly to the Chief AI Officer, tasked with establishing and operationalizing a global framework that balances breakthrough AI innovation with robust governance and unwavering ethical standards.

    You will lead a global team to build the essential guardrails that enable our scientists, clinicians, and business leaders to leverage AI confidently and compliantly. Your group will serve as the central hub of expertise on AI ethics, risk management, and regulatory compliance, empowering teams across our Pharmaceuticals and Diagnostics divisions.

    The Opportunity:

    Risk & Ethics Framework: Design, implement, and lead the global framework for managing AI-specific risks. This includes establishing the ethical principles, policies, and control mechanisms that ensure responsible AI development and deployment, in alignment with the overall AI strategy
    Risk Management Framework: Design and operate a comprehensive risk management framework to identify, assess, and mitigate ethical, legal, security, and reputational risks associated with AI systems, from research to deployment
    Strategic Partnership & Consulting: Act as a primary strategic partner to key leaders, especially the Head of AI Strategy & Governance, to embed ethical and risk-based considerations directly into the core AI lifecycle. Provide expert consultation to development teams on mitigating risks such as bias, lack of transparency, and privacy violations
    Regulatory Leadership: Monitor the global regulatory landscape for AI (e.g., EU AI Act, FDA/EMA guidelines) and translate complex requirements into actionable corporate policies and practical guidance for technical and non-technical audiences.
    Ethical Oversight & Review: Establish and chair an AI Ethics Board or review council to provide oversight for high-risk AI use cases, ensuring alignment with Roche’s values and patient-centric mission
    Leadership & Team Development: Lead and mentor a high-performing global team of AI risk and ethics specialists, fostering a culture of expertise, collaboration, and pragmatic problem-solving
    Culture & Enablement: Champion a culture of responsible AI innovation. Develop and deliver training and awareness programs to build AI literacy and ethical decision-making capabilities across the organization
    GxP and Patient Data Compliance: Ensure that all AI governance frameworks and solutions deployed in regulated areas are fully compliant with GxP standards (e.g., 21 CFR Part 11) and global patient data privacy regulations (e.g., GDPR, HIPAA)


    Who you are:

    Education: Bachelor’s or Master’s degree in a relevant field such as Computer Science, Law, Information Systems, Bioethics, or a related discipline. A combination of technical and legal/ethical education is highly advantageous.
    Experience:
    A minimum of 10+ years of overall professional experience in technology, governance, or risk management within a large, global, matrixed organization
    A mandatory 5 years of direct, specialized experience in AI ethics, AI governance, or technology risk management
    Pharmaceutical or life sciences industry experience is strongly preferred to ensure credibility and ability to navigate the specific challenges of our industry.
    Leadership & Influence:
    Proven experience leading and developing global teams, with a track record of managing both direct and indirect reports
    Exceptional influencing skills, with the ability to build consensus and drive alignment among senior stakeholders with diverse priorities.
    Technical & Regulatory Knowledge:
    Deep understanding of AI/ML concepts, model development lifecycles, and associated risks (e.g., bias, transparency, security)
    Expert knowledge of the global AI regulatory environment, including the EU AI Act.
    Core Competencies:
    Demonstrated executive presence and exceptional communication skills.
    A pragmatic problem-solver who can create effective controls without stifling innovation
    High tolerance for navigating ambiguity and the ability to lead effectively in a rapidly evolving field

    Who we are: 
    A healthier future drives us to innovate. Together, more than 100’000 employees across the globe are dedicated to advance science, ensuring everyone has access to healthcare today and for generations to come. Our efforts result in more than 26 million people treated with our medicines and over 30 billion tests conducted using our Diagnostics products. We empower each other to explore new possibilities, foster creativity, and keep our ambitions high, so we can deliver life-changing healthcare solutions that make a global impact.

"""

# ---------- 2) LLM chain (robust prompt) ----------

parser = PydanticOutputParser(pydantic_object=MatchResult)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior technical recruiter and hiring manager with 15+ years "
                "experience in software, data, and machine learning roles.\n\n"
                "The job and CV JSON structures can vary a lot: different field names, "
                "free-text descriptions, nested sections, etc. Your job is to:\n"
                "1) Carefully understand the job description JSON (title, text, bullet points, "
                "requirements, responsibilities, etc.).\n"
                "2) Carefully understand the candidate CV JSON (summary, roles, tech stack, "
                "projects, education, etc.).\n"
                "3) Focus on the MOST IMPORTANT information only: core responsibilities, "
                "must-have skills, seniority, main tech stack, domains, leadership.\n"
                "   Ignore noise like hobbies unless they clearly support the role.\n"
                "4) Build a structured candidate_profile from whatever useful signals you find.\n"
                "5) Extract a clear list of requirements from the job (extracted_requirements):\n"
                "   - For each requirement, set:\n"
                "     * name (short requirement name)\n"
                "     * priority (e.g. 'must_have', 'nice_to_have', 'optional')\n"
                "     * category (e.g. 'skill', 'soft_skill', 'experience', "
                "'domain_knowledge', 'tool', 'other')\n"
                "     * details (short explanation in context)\n"
                "6) For each requirement, classify it into exactly one bucket:\n"
                "   - explicit_matches: clearly and explicitly present in the CV.\n"
                "   - obvious_gaps: clearly missing or unlikely from the CV.\n"
                "   - inferred_matches: not explicitly written, but strongly suggested by "
                "     education, frameworks, domains, tools, or very typical background.\n\n"
                "Reasoning rules:\n"
                "- Be strict for explicit_matches: require clear textual evidence in the CV.\n"
                "- For inferred_matches, only infer when there is strong contextual support.\n"
                "- Use obvious_gaps for important requirements that are absent or unlikely.\n"
                "- Always provide a confidence score in [0,1] for each item.\n\n"
                "Return ONLY JSON that matches the given schema.\n"
                "{format_instructions}"
            ),
        ),
        (
            "user",
            "Job description JSON:\n{job_json}\n\nCandidate CV JSON:\n{cv_json}"
        ),
    ]
)

# llm = ChatOpenAI(
#     model="gpt-5-nano",   # or "gpt-4.1" for stronger reasoning       # more deterministic, recruiter-like
# )

# chain = prompt | llm | parser


# ---------- 3) Public helpers ----------

def match_job_and_cv_from_dicts(job_json: dict, cv_json: dict) -> MatchResult:
    return chain.invoke(
        {
            "job_json": job_json,
            "cv_json": cv_json,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def match_job_and_cv_from_files(job_path: str, cv_path: str) -> MatchResult:
    with open(job_path, "r", encoding="utf-8") as f:
        job_json = json.load(f)
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_json = json.load(f)
    return match_job_and_cv_from_dicts(job_json, cv_json)


def get_llm():
    llm =ChatGroq(model="llama-3.3-70b-versatile", groq_api_key= GROQ_API_KEY, temperature=1)
    return llm

# ---------- Utility to load PDF text ----------
def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = " ".join([page.page_content for page in pages])
    return text

# ---------- PDF to JSON conversion utility ----------
def convert_pdf_to_json(pdf_path):
    try:
        if not pdf_path or not os.path.isfile(pdf_path):
            return {"message": "Provide the resume file"}
        pdf_text_data = load_pdf_text(pdf_path)
        gemma_model = get_llm()
        resume_text_to_json_chain = RESUME_TEXT_TO_JSON_PROMPT | gemma_model | JsonOutputParser()
        resume_json_data = resume_text_to_json_chain.invoke({"resume_text": pdf_text_data})
        print(f"JSON DATA: {resume_json_data}")
        return resume_json_data
    except Exception as e:
        raise Exception("Unable to process the resume file due to : " + str(e))


# ---------- Main matching function ----------
def match_job_and_resume(job_description, resume_file_path):
    try:
        parser = PydanticOutputParser(pydantic_object=MatchResult)
        start_time = datetime.now()
        print(f"start time: {start_time}")
        resume_json_data = convert_pdf_to_json(resume_file_path)
        gemma_model = get_llm()
        match_job_and_resume_chain = MATHCH_JOB_AND_RESUME_PROMPT | gemma_model | parser
        result = match_job_and_resume_chain.invoke({
            "job_text_data": job_description,
            "resume_json": resume_json_data,
            "format_instructions": parser.get_format_instructions()
        })
        json_result = json.loads(result.model_dump_json(indent=2))
        print(f"=============== Match Result ===============")
        print(f"Match Result: {json_result}")
        print(f"Type of result {type(json_result)}")
        end_time = datetime.now()
        print(f"end time: {end_time}")

        return json_result
    except Exception as e:
        print(f"Error : {str(e)}")
        return None 


# ---------- 4) CLI / quick test ----------

if __name__ == "__main__":
    # if not os.getenv("OPENAI_API_KEY"):
    #     raise RuntimeError(
    #         "Please set the OPENAI_API_KEY environment variable before running."
    #     )
    result = match_job_and_resume(job_description, "Nadeem_Full_Stack_Engineer_Resume.pdf")

