import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:4b"

def check_ollama_status():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running.")
            return True
        else:
            print(f"❌ Ollama server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server. Make sure it is running.")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection to Ollama server timed out.")
        return False

def check_model_available():
    """Check if the qwen3:4b model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if any(MODEL_NAME in name for name in model_names):
                print(f"✅ Model '{MODEL_NAME}' is available.")
                return True
            else:
                print(f"❌ Model '{MODEL_NAME}' is NOT available.")
                print(f"   Available models: {', '.join(model_names) if model_names else 'None'}")
                print(f"   Run: ollama pull {MODEL_NAME}")
                return False
    except Exception as e:
        print(f"❌ Error checking model availability: {e}")
        return False

def test_model_inference():
    """Test a simple inference with qwen3:4b."""
    print(f"\n🔄 Testing inference with '{MODEL_NAME}'...")
    payload = {
        "model": MODEL_NAME,
        "prompt": "Say 'Hello, configuration test successful!' and nothing else.",
        "stream": False
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            output = result.get("response", "").strip()
            print(f"✅ Model response: {output}")
            return True
        else:
            print(f"❌ Inference failed with status code: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Inference request timed out.")
        return False
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def print_model_info():
    """Print detailed information about the qwen3:4b model."""
    try:
        payload = {"name": MODEL_NAME}
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            info = response.json()
            print(f"\n📋 Model Information for '{MODEL_NAME}':")
            print(f"   Parameters: {info.get('details', {}).get('parameter_size', 'N/A')}")
            print(f"   Quantization: {info.get('details', {}).get('quantization_level', 'N/A')}")
            print(f"   Format: {info.get('details', {}).get('format', 'N/A')}")
            print(f"   Family: {info.get('details', {}).get('family', 'N/A')}")
        else:
            print(f"⚠️  Could not retrieve model info (status: {response.status_code})")
    except Exception as e:
        print(f"⚠️  Error retrieving model info: {e}")

def main():
    print("=" * 50)
    print("  Ollama Configuration Check")
    print(f"  Model: {MODEL_NAME}")
    print(f"  URL:   {OLLAMA_BASE_URL}")
    print("=" * 50)

    server_ok = check_ollama_status()
    if not server_ok:
        print("\n❌ Configuration check failed. Ollama server is not reachable.")
        return

    model_ok = check_model_available()
    if not model_ok:
        print("\n❌ Configuration check failed. Required model is not available.")
        return

    print_model_info()
    inference_ok = test_model_inference()

    print("\n" + "=" * 50)
    if server_ok and model_ok and inference_ok:
        print("✅ All checks passed! Ollama is configured correctly.")
    else:
        print("❌ Some checks failed. Please review the output above.")
    print("=" * 50)

if __name__ == "__main__":
    main()