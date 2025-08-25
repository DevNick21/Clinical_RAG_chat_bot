"""
Compatibility shim for huggingface-hub API changes.
This should be imported before any sentence-transformers imports.
"""


def patch_huggingface_hub():
    """
    Patches huggingface-hub to provide backward compatibility for cached_download.

    This fixes the issue where sentence-transformers expects cached_download
    but newer versions of huggingface-hub only provide hf_hub_download.
    """
    try:
        import huggingface_hub

        # Only patch if cached_download is missing
        if hasattr(huggingface_hub, 'cached_download'):
            print("✓ cached_download already available, no patching needed")
            return

        if not hasattr(huggingface_hub, 'hf_hub_download'):
            print("✗ Neither cached_download nor hf_hub_download available")
            return

        from huggingface_hub import hf_hub_download
        import inspect

        # Get the signature of the original hf_hub_download to understand parameters
        def cached_download(*args, **kwargs):
            """
            Compatibility wrapper that maps cached_download calls to hf_hub_download.

            This handles the most common usage patterns in sentence-transformers.
            """
            # Handle different calling conventions
            if len(args) == 1 and isinstance(args[0], str):
                # Most common case: cached_download(url, **other_params)
                url_or_repo_id = args[0]

                # Extract repo_id and filename from various formats
                if url_or_repo_id.startswith('http'):
                    # It's a URL - try to extract repo and file info
                    # This is a simplified parser - might need refinement
                    parts = url_or_repo_id.replace(
                        'https://huggingface.co/', '').split('/')
                    if len(parts) >= 3 and parts[-2] == 'resolve':
                        repo_id = '/'.join(parts[:-3])
                        filename = parts[-1]
                    else:
                        # Fallback handling
                        repo_id = kwargs.get('repo_id', url_or_repo_id)
                        filename = kwargs.get('filename', 'config.json')
                else:
                    # Assume it's already a repo_id
                    repo_id = url_or_repo_id
                    filename = kwargs.get('filename', 'config.json')

                # Remove filename from kwargs if it exists to avoid duplicate parameter
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('filename', None)
                kwargs_copy.pop('repo_id', None)

                # Handle parameter name changes
                if 'use_auth_token' in kwargs_copy:
                    kwargs_copy['token'] = kwargs_copy.pop('use_auth_token')

                return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs_copy)

            else:
                # Pass through with parameter name mapping
                kwargs_copy = kwargs.copy()
                if 'use_auth_token' in kwargs_copy:
                    kwargs_copy['token'] = kwargs_copy.pop('use_auth_token')

                return hf_hub_download(*args, **kwargs_copy)

        # Add the compatibility function
        huggingface_hub.cached_download = cached_download
        print("✓ Applied huggingface-hub compatibility patch: cached_download -> hf_hub_download")

    except Exception as e:
        print(f" Could not apply huggingface-hub compatibility patch: {e}")


# Apply the patch when this module is imported
patch_huggingface_hub()
