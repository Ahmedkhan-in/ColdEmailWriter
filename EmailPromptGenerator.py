from langchain_core.prompts import PromptTemplate
prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Ahmed, a business development executive at Betra. Betra is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Betra 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Betra's portfolio: {link_list}
        Remember you are Mohan, BDE at Betra. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )
# Save the template for later use
prompt_email.save("email_prompt.json")