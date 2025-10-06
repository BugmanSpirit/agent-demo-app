"""Modal deployment configuration for Web Search Agent."""

import modal
from pathlib import Path

# Project root path
ROOT_DIR = Path(__file__).parent

# Create Modal app
app = modal.App("parallel-search-agent")

# Define the image with dependencies from pyproject.toml and bundle static assets
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_file(ROOT_DIR / "index.html", "/assets/index.html")
)

# Create secrets for API keys
# Run: modal secret create parallel-search-secrets PARALLEL_API_KEY=your_key GEMINI_API_KEY=your_key


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("parallel-search-secrets")],
    min_containers=1,  # Keep one instance warm
    timeout=300,  # 5 minute timeout for long searches
)
@modal.asgi_app()
def fastapi_app():
    """Create and return the FastAPI application with all endpoints."""
    import os
    import json
    from datetime import datetime
    from typing import AsyncGenerator, Optional, Dict, Any, AsyncIterator
    
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import google.generativeai as genai
    import httpx
    
    # Constants
    PARALLEL_API_URL = "https://api.parallel.ai/v1beta/search"
    GEMINI_MODEL = "gemini-2.5-flash"
    MAX_SEARCH_STEPS = 25
    
    # Prompts
    SEARCH_TOOL_DESCRIPTION = """# Web Search Tool

**Purpose:** Perform web searches and return LLM-friendly results.

**Usage:**
- objective: Natural-language description of your research goal (max 200 characters)

**Best Practices:**
- Be specific about what information you need
- Mention if you want recent/current data
- Keep objectives concise but descriptive"""
    
    SEARCH_OBJECTIVE_DESCRIPTION = "Natural-language description of your research goal (max 200 characters)"
    
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a simple search agent. Your mission is to comprehensively fulfill the user's search objective by conducting 1 up to 3 searches from different angles until you have gathered sufficient information to provide a complete answer. The current date is {current_date}

**Research Philosophy:**
- Each search should explore a unique angle or aspect of the topic
- NEVER try to OPEN an article, the excerpts provided should be enough

**Key Parameters:**
- objective: Describe what you're trying to accomplish. This helps the search engine understand intent and provide relevant results.

**Output:**
After doing the searches required, write up your 'search report' that answers the initial search query. Even if you could not answer the question ensure to always provide a final report! Please do NOT use markdown tables. """
    
    # Get API keys from Modal secrets (environment variables)
    PARALLEL_API_KEY = os.environ.get("PARALLEL_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    # Configure Gemini
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize FastAPI app
    web_app = FastAPI(
        title="Web Search Agent",
        description="AI-powered web search using Google Gemini and Parallel Search API",
        version="1.0.0"
    )
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Models
    class SearchRequest(BaseModel):
        """Request model for search endpoint."""
        query: str = Field(..., description="User's search query")
        systemPrompt: Optional[str] = Field(None, description="Optional custom system prompt")
    
    # Helper functions
    def search_web(objective: str) -> Dict[str, Any]:
        """Perform web search using Parallel API via REST."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": PARALLEL_API_KEY
        }
        payload = {
            "objective": objective,
            "processor": "base",
            "max_results": 10,
            "max_chars_per_result": 2500
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(PARALLEL_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    
    def create_search_tool() -> genai.protos.Tool:
        """Create the search tool declaration for Gemini function calling."""
        return genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="search",
                    description=SEARCH_TOOL_DESCRIPTION,
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "objective": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description=SEARCH_OBJECTIVE_DESCRIPTION
                            )
                        },
                        required=["objective"]
                    )
                )
            ]
        )
    
    def get_default_system_prompt() -> str:
        """Get the default system prompt for the search agent."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date)
    
    async def process_response_stream(chat: Any, function_response: Any, initial_step_count: int) -> AsyncGenerator[str, None]:
        """Process a response stream from Gemini after a function call."""
        import asyncio
        
        step_count = initial_step_count
        next_response = await asyncio.to_thread(lambda: chat.send_message(function_response, stream=True))
        
        for next_chunk in next_response:
            step_count += 1
            if step_count >= MAX_SEARCH_STEPS:
                yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'max_steps'})}\n\n"
                break
                
            if next_chunk.candidates and next_chunk.candidates[0].content.parts:
                for next_part in next_chunk.candidates[0].content.parts:
                    if hasattr(next_part, 'text') and next_part.text:
                        yield f"data: {json.dumps({'type': 'text-delta', 'text': next_part.text})}\n\n"
                        
                    elif hasattr(next_part, 'function_call') and next_part.function_call:
                        fc2 = next_part.function_call
                        yield f"data: {json.dumps({'type': 'tool-call', 'toolName': fc2.name, 'args': dict(fc2.args)})}\n\n"
                        
                        if fc2.name == "search":
                            objective2 = fc2.args.get("objective", "")
                            search_result2 = search_web(objective2)
                            yield f"data: {json.dumps({'type': 'tool-result', 'toolName': 'search', 'output': search_result2})}\n\n"
                            
                            function_response2 = genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=fc2.name,
                                    response={"result": search_result2}
                                )
                            )
                            
                            async for item in process_response_stream(chat, function_response2, step_count):
                                yield item
    
    async def stream_research(query: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream research results using Gemini with function calling."""
        import asyncio
        
        system_instruction = system_prompt or get_default_system_prompt()
        
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=[create_search_tool()],
            system_instruction=system_instruction
        )
        
        chat = model.start_chat(enable_automatic_function_calling=False)
        step_count = 0
        
        try:
            response = await asyncio.to_thread(lambda: chat.send_message(query, stream=True))
            
            for chunk in response:
                step_count += 1
                
                if step_count >= MAX_SEARCH_STEPS:
                    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'max_steps'})}\n\n"
                    break
                
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            yield f"data: {json.dumps({'type': 'text-delta', 'text': part.text})}\n\n"
                        
                        elif hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            yield f"data: {json.dumps({'type': 'tool-call', 'toolName': fc.name, 'args': dict(fc.args)})}\n\n"
                            
                            if fc.name == "search":
                                objective = fc.args.get("objective", "")
                                search_result = search_web(objective)
                                yield f"data: {json.dumps({'type': 'tool-result', 'toolName': 'search', 'output': search_result})}\n\n"
                                
                                function_response = genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=fc.name,
                                        response={"result": search_result}
                                    )
                                )
                                
                                async for item in process_response_stream(chat, function_response, step_count):
                                    yield item
            
            finish_reason = (
                str(response.candidates[0].finish_reason) 
                if response.candidates and response.candidates[0].finish_reason 
                else 'completed'
            )
            yield f"data: {json.dumps({'type': 'finish', 'finishReason': finish_reason})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    # API Endpoints
    @web_app.get("/", response_class=HTMLResponse)
    async def serve_index() -> str:
        """Serve the HTML frontend copied into the Modal image."""
        index_asset_path = Path("/assets/index.html")
        if not index_asset_path.exists():
            return "<html><body><h1>index.html not bundled</h1></body></html>"
        return index_asset_path.read_text(encoding="utf-8")
    
    @web_app.post("/agents/cerebras-search")
    async def research_endpoint(request: SearchRequest) -> StreamingResponse:
        """Handle research requests with streaming SSE response."""
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        if not PARALLEL_API_KEY or not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Server configuration error: Missing API keys"
            )
        
        async def event_generator() -> AsyncIterator[str]:
            async for chunk in stream_research(request.query, request.systemPrompt):
                yield chunk
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )
    
    @web_app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "gemini_configured": "yes" if GEMINI_API_KEY else "no",
            "parallel_configured": "yes" if PARALLEL_API_KEY else "no"
        }
    
    return web_app
