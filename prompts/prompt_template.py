from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


RESUME_TEXT_TO_JSON_PROMPT = PromptTemplate(
    input_variables=["resume_text"],
    template="""
        You are an expert Resume Parser AI. 
        Extract all below relevant fields information from the given resume text and return only JSON object by folowing below rules.

        Fields to extract:
        - current_location
        - skills
        - education (degree, institution, year)
        - experience (job_title, company, start_date, end_date, description)    
        - certifications
        - projects
        - languages
        - Training & Workshops
        - Achievements & Awards
        - Publications
        

        Rules:
        1. Output exactly in the provided JSON format.
        2. If a fields is missing in text, do not include in JSON output.
        3. Normalize dates consistently (e.g., "Jan 2020", "January 2020").
        4. Rremove duplicates.
        6. Ensure the JSON is valid and properly formatted.
        7. Do NOT include any explanations, quotes, markdown, or code fences and words outside the givent text data

        Input Resume Text:
        {resume_text}
    """
)

MATHCH_JOB_AND_RESUME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior technical recruiter and hiring manager with 15+ years "
                "experience in software, data, and machine learning roles.\n\n"
                "The job and CV JSON structures can vary a lot: different field names, "
                "free-text descriptions, nested sections, etc. Your job is to:\n"
                "1) Carefully understand the job description data.\n"
                "2) Carefully understand the candidate CV JSON (Experience, skills, "
                "projects, education, etc.).\n"
                "3) Focus on the MOST IMPORTANT information only: core responsibilities, "
                "must-have skills, seniority, main tech stack, domains, leadership.\n"
                "   Ignore noise like hobbies unless they clearly support the role.\n"
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
            "Job description text data:\n{job_text_data}\n\nCandidate CV JSON:\n{resume_json}"
        ),
    ]
)
