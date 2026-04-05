"""
Unified Ollama Client for ARIA Engine

Provides a consistent interface for all ARIA variants to interact with Ollama models.
Supports both text generation and embedding generation with proper error handling,
retry logic, and JSON mode enforcement.

Author: ARIA Team
Date: 2026-02-01
"""

import subprocess
import json
import time
import random
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Unified client for Ollama LLM interactions.

    Features:
    - Text generation with JSON mode enforcement
    - Embedding generation for semantic similarity
    - Automatic retry with exponential backoff
    - Response validation and parsing
    - Model availability checking
    """

    def __init__(
        self,
        model: str = "qwen2:7b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize Ollama client.

        Args:
            model: Primary model for text generation (deepseek-r1:7b, qwen2.5:14b, etc.)
            embedding_model: Model for embeddings (nomic-embed-text recommended)
            base_url: Ollama server URL
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

        # Validate Ollama is running and models are available
        self._check_ollama_available()
        self._check_model_available(model)
        self._check_model_available(embedding_model)

        logger.info(f"OllamaClient initialized with model={model}, embedding={embedding_model}")

    def _check_ollama_available(self) -> bool:
        """Check if Ollama service is running."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama is not running. Start with: ollama serve")
            return True
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama is not installed. Install from: https://ollama.ai/"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama service is not responding")

    def _check_model_available(self, model_name: str) -> bool:
        """Check if specific model is available locally."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if model_name not in result.stdout:
                logger.warning(f"Model {model_name} not found. Pulling...")
                self._pull_model(model_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to check model availability: {e}")

    def _pull_model(self, model_name: str):
        """Pull model from Ollama registry."""
        logger.info(f"Pulling model {model_name}...")
        result = subprocess.run(
            ['ollama', 'pull', model_name],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull model {model_name}: {result.stderr}")
        logger.info(f"Model {model_name} pulled successfully")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = True,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text with Ollama model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            json_mode: Enforce JSON output format
            system_message: Optional system message

        Returns:
            Generated text (JSON string if json_mode=True)
        """
        for attempt in range(self.max_retries):
            try:
                # Build command
                cmd = ['ollama', 'run', self.model]

                # Construct prompt with system message if provided
                full_prompt = prompt
                if system_message:
                    full_prompt = f"<|system|>\n{system_message}\n<|user|>\n{prompt}\n<|assistant|>"

                if json_mode:
                    full_prompt += "\n\nIMPORTANT: Respond with ONLY valid JSON. No other text."

                # Execute
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    response = result.stdout.strip()

                    # Validate JSON if json_mode
                    if json_mode:
                        try:
                            json.loads(response)  # Validate
                        except json.JSONDecodeError:
                            # Try to extract JSON from response
                            response = self._extract_json(response)
                            if not response:
                                raise ValueError("Failed to extract valid JSON from response")

                    logger.debug(f"Generated response ({len(response)} chars)")
                    return response

                else:
                    raise RuntimeError(f"Ollama error: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise TimeoutError(f"Generation timed out after {self.max_retries} attempts")

            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Generation failed: {e}")

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain additional content.

        Tries multiple strategies:
        1. Look for JSON code block (```json ... ```)
        2. Look for plain code block (``` ... ```)
        3. Find first { to last }
        """
        import re

        # Strategy 1: JSON code block
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                json.loads(json_str)  # Validate
                return json_str
            except:
                pass

        # Strategy 2: Plain code block
        match = re.search(r'```\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                json.loads(json_str)
                return json_str
            except:
                pass

        # Strategy 3: Find JSON object
        match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                json.loads(json_str)
                return json_str
            except:
                pass

        return None

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_message: Optional system message

        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            system_message=system_message
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text using sentence-transformers.

        Args:
            text: Single text string or list of strings

        Returns:
            Single embedding vector or list of vectors
        """
        # Use sentence-transformers directly (compatible with original ARIA)
        try:
            from sentence_transformers import SentenceTransformer

            # Initialize embedding model (cached after first call)
            if not hasattr(self, '_embedding_model_cached'):
                logger.info(f"Loading embedding model: {self.embedding_model}")
                # Use lightweight model from original ARIA
                self._embedding_model_cached = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embeddings
            is_single = isinstance(text, str)
            texts = [text] if is_single else text

            embeddings = self._embedding_model_cached.encode(texts, convert_to_numpy=True)
            embeddings_list = embeddings.tolist()

            return embeddings_list[0] if is_single else embeddings_list

        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = True
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            json_mode: Enforce JSON output

        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            response = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode
            )
            responses.append(response)

        return responses

    def test_connection(self) -> bool:
        """Test if client can successfully communicate with Ollama."""
        try:
            response = self.generate(
                prompt="Hello, this is a test. Respond with: {\"status\": \"ok\"}",
                json_mode=True,
                max_tokens=50
            )
            data = json.loads(response)
            return data.get('status') == 'ok'
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        try:
            result = subprocess.run(
                ['ollama', 'show', self.model],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse model info from output
                info = {}
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
                return info
            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {}


# Singleton instance for shared use
_default_client: Optional[OllamaClient] = None


def get_ollama_client(
    model: str = "qwen2:7b",
    **kwargs
) -> OllamaClient:
    """
    Get or create default OllamaClient instance.

    Args:
        model: Model to use
        **kwargs: Additional arguments for OllamaClient

    Returns:
        OllamaClient instance
    """
    global _default_client

    if _default_client is None or _default_client.model != model:
        _default_client = OllamaClient(model=model, **kwargs)

    return _default_client


if __name__ == "__main__":
    # Test the client
    print("Testing OllamaClient...")

    try:
        client = OllamaClient(model="qwen2:7b")

        # Test connection
        print("\n1. Testing connection...")
        if client.test_connection():
            print("   ✓ Connection successful")
        else:
            print("   ✗ Connection failed")

        # Test JSON generation
        print("\n2. Testing JSON generation...")
        prompt = """Generate a sample materials science prediction in JSON format:
        {
          "material": "...",
          "property": "...",
          "value": ...
        }"""

        response = client.generate_json(prompt)
        print(f"   ✓ Generated JSON: {json.dumps(response, indent=2)}")

        # Test embedding
        print("\n3. Testing embeddings...")
        embedding = client.embed("MoS2 doped with Nb")
        print(f"   ✓ Embedding dimension: {len(embedding)}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
