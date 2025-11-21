#!/usr/bin/env python3
"""
Script to clone all published repos from the COMPUTO organization.
Published repos have names that start with "published-" followed by a date.
"""

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from github import Github
from tqdm import tqdm


def parse_repo_name(repo_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse a published repo name to extract date and metadata.
    
    Args:
        repo_name: Repo name like "published-202510-durand-fast"
    
    Returns:
        Tuple of (date, metadata) like ("2025-10", "durand-fast"), or None if not a published repo
    
    Examples:
        >>> parse_repo_name("published-202510-durand-fast")
        ("2025-10", "durand-fast")
        >>> parse_repo_name("published-202306-sanou-multiscale_glasso")
        ("2023-06", "sanou-multiscale_glasso")
        >>> parse_repo_name("template-computo-julia")
        None
    """
    if not repo_name.startswith("published-"):
        return None
    
    # Remove "published-" prefix
    parts = repo_name[len("published-"):]
    
    # Split by first hyphen after the date (YYYYMM format)
    if len(parts) < 6:  # Need at least YYYYMM
        return None
    
    # Extract YYYYMM
    date_str = parts[:6]
    if not date_str.isdigit():
        return None
    
    # Format as YYYY-MM
    year = date_str[:4]
    month = date_str[4:6]
    formatted_date = f"{year}-{month}"
    
    # Everything after "published-YYYYMM-" is metadata
    if len(parts) > 6 and parts[6] == "-":
        metadata = parts[7:]
    else:
        metadata = ""
    
    return formatted_date, metadata


def load_existing_repos(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load existing repos from CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Dictionary mapping repo_name to repo data
    """
    if not csv_path.exists():
        return {}
    
    repos = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            repos[row["repo_name"]] = {
                "repo_url": row["repo_url"],
                "date": row["date"],
                "metadata": row["metadata"]
            }
    except Exception as e:
        print(f"Warning: Could not load existing CSV: {e}")
        return {}
    
    return repos


def clone_repo(repo_url: str, target_path: Path) -> bool:
    """
    Clone a git repository to the target path.
    
    Args:
        repo_url: URL of the repository to clone
        target_path: Directory where the repo should be cloned
    
    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "clone", repo_url, str(target_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError cloning {repo_url}: {e.stderr}")
        return False
    except Exception as e:
        print(f"\nError cloning {repo_url}: {e}")
        return False


def main():
    """Main function to clone all published repos from COMPUTO organization."""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "documents" / "computo"
    csv_path = output_dir / "computo_repos.csv"
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GitHub client (no authentication)
    print("Connecting to GitHub...")
    g = Github()
    
    try:
        # Get the COMPUTO organization
        org = g.get_organization("computorg")
        
        # Get all repositories
        print("Fetching repositories from computorg organization...")
        all_repos = list(org.get_repos(type="public"))
        print(f"Found {len(all_repos)} total repositories")
        
        # Filter for published repos
        published_repos = []
        for repo in all_repos:
            parsed = parse_repo_name(repo.name)
            if parsed is not None:
                date, metadata = parsed
                published_repos.append({
                    "name": repo.name,
                    "url": repo.clone_url,
                    "date": date,
                    "metadata": metadata
                })
        
        print(f"Found {len(published_repos)} published repositories")
        
        # Load existing repos
        existing_repos = load_existing_repos(csv_path)
        print(f"Already have {len(existing_repos)} repos in CSV")
        
        # Determine which repos need to be cloned
        repos_to_clone = []
        for repo_data in published_repos:
            if repo_data["name"] not in existing_repos:
                repos_to_clone.append(repo_data)
        
        print(f"Need to clone {len(repos_to_clone)} new repositories")
        
        if len(repos_to_clone) == 0:
            print("All repos are already cloned!")
            return
        
        # Clone repos with progress bar
        all_repos_data = list(existing_repos.values()) if existing_repos else []
        
        with tqdm(total=len(repos_to_clone), desc="Cloning repos") as pbar:
            for repo_data in repos_to_clone:
                repo_name = repo_data["name"]
                repo_url = repo_data["url"]
                target_path = output_dir / repo_name
                
                pbar.set_description(f"Cloning {repo_name}")
                
                # Clone the repo
                if clone_repo(repo_url, target_path):
                    # Add to our data list
                    all_repos_data.append({
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "date": repo_data["date"],
                        "metadata": repo_data["metadata"]
                    })
                    pbar.set_postfix({"status": "success"})
                else:
                    pbar.set_postfix({"status": "failed"})
                
                pbar.update(1)
        
        # Save updated CSV
        print("\nSaving CSV file...")
        df = pd.DataFrame(all_repos_data)
        df = df.sort_values(by="date", ascending=False)  # Sort by date, newest first
        df.to_csv(csv_path, index=False)
        
        print(f"\nDone! CSV saved to {csv_path}")
        print(f"Total repos tracked: {len(all_repos_data)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

