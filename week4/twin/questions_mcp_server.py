from mcp.server.fastmcp import FastMCP
import questions

mcp = FastMCP("questions_server")


@mcp.tool()
async def get_questions_with_answer() -> str:
    """
    Retrieve from the database all the recorded questions where you have been provided with an official answer.

    Returns:
        A string containing the questions with their official answers.
    """
    return questions.get_questions_with_answer()

@mcp.tool()
async def record_question_with_no_answer(question: str) -> str:
    """
    Record the question into the database without an official answer.
    
    Returns:
        Confirmation of which question was recorded into the dabase without an answer.
    """
    return questions.record_question_with_no_answer(question)

@mcp.tool()
async def get_questions_with_no_answer() -> str:
    """
    Retrieve from the database all the recorded questions without an official answer.
    
    Returns:
        A string containing the questions that do no have official answer.
    """
    return questions.get_questions_with_no_answer()

@mcp.tool()
async def record_answer_to_question(id: int, answer: str) -> str:
    """
    Record the answer into the database for the question with the provided id.
    
    Returns:
        Confirmation of which question was in the database was updated with the answer.
    """
    return questions.record_answer_to_question(id, answer)



if __name__ == "__main__":
    mcp.run(transport="stdio")
