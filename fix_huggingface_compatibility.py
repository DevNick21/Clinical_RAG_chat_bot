"""
Compatibility fix for huggingface-hub API changes
This fixes the issue where sentence-transformers tries to import 'cached_download'
which has been replaced by 'hf_hub_download' in newer versions of huggingface-hub.
"""


def apply_huggingface_compatibility_fix():
    """Apply compatibility fix for huggingface-hub API changes"""
    try:
        import huggingface_hub

        # Check if cached_download is missing (which means we need the fix)
        if not hasattr(huggingface_hub, 'cached_download'):
            # Import the new function
            from huggingface_hub import hf_hub_download

            # Create a wrapper function that matches the old cached_download signature
            def cached_download(url, **kwargs):
                """Compatibility wrapper for the old cached_download function"""
                # Extract the repo_id and filename from the URL or kwargs
                # The old cached_download used different parameter names

                # Handle both old and new parameter styles
                if 'repo_id' in kwargs and 'filename' in kwargs:
                    # New style parameters
                    return hf_hub_download(**kwargs)
                else:
                    # Try to parse old-style URL-based calls
                    # This is a simplified version - you may need to adjust based on actual usage
                    return hf_hub_download(url, **kwargs)

            # Monkey patch the old function name
            huggingface_hub.cached_download = cached_download

            print(
                "✓ Applied huggingface-hub compatibility fix: cached_download -> hf_hub_download")

    except ImportError as e:
        print(
            f"Warning: Could not apply huggingface-hub compatibility fix: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error applying compatibility fix: {e}")


# Apply the fix immediately when this module is imported
if __name__ != "__main__":
    apply_huggingface_compatibility_fix()
