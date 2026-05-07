#!/bin/bash

# RAG Assistant - GitHub Setup and Push Script
# This script helps you set up and push the project to GitHub

set -e  # Exit on error

echo "=========================================="
echo "Conversation Agent - GitHub Setup"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if git is installed
echo -e "${YELLOW}[1/5] Checking git installation...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git is not installed. Please install it first.${NC}"
    echo "On macOS: brew install git"
    echo "On Ubuntu: sudo apt-get install git"
    exit 1
fi
echo -e "${GREEN}✓ Git is installed${NC}"
echo ""

# Step 2: Check git configuration
echo -e "${YELLOW}[2/5] Checking git configuration...${NC}"
git_user=$(git config --global user.name 2>/dev/null || echo "")
git_email=$(git config --global user.email 2>/dev/null || echo "")

if [ -z "$git_user" ] || [ -z "$git_email" ]; then
    echo -e "${YELLOW}Git user configuration not found.${NC}"
    read -p "Enter your GitHub username: " github_user
    read -p "Enter your GitHub email: " github_email
    
    git config --global user.name "$github_user"
    git config --global user.email "$github_email"
    echo -e "${GREEN}✓ Git configured with user: $github_user${NC}"
else
    echo -e "${GREEN}✓ Git configured with user: $git_user ($git_email)${NC}"
fi
echo ""

# Step 3: Initialize git repository
echo -e "${YELLOW}[3/5] Setting up git repository...${NC}"
if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Git repository already initialized${NC}"
else
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi
echo ""

# Step 4: Check for remote
echo -e "${YELLOW}[4/5] Checking git remote...${NC}"
if git remote get-url origin &> /dev/null; then
    current_remote=$(git remote get-url origin)
    echo -e "${GREEN}✓ Remote already configured: $current_remote${NC}"
else
    echo -e "${YELLOW}No remote repository configured yet.${NC}"
    echo ""
    echo "You have two options for GitHub authentication:"
    echo "1) HTTPS with Personal Access Token (recommended for beginners)"
    echo "2) SSH (requires SSH key setup)"
    echo ""
    read -p "Choose option (1 or 2): " auth_choice
    
    read -p "Enter your GitHub username: " github_user
    read -p "Enter your GitHub repository name: " repo_name
    
    if [ "$auth_choice" = "2" ]; then
        # SSH
        remote_url="git@github.com:${github_user}/${repo_name}.git"
        echo ""
        echo -e "${YELLOW}Using SSH authentication${NC}"
        echo "Make sure you have SSH keys set up:"
        echo "  - Keys location: ~/.ssh/id_rsa (private) and ~/.ssh/id_rsa.pub (public)"
        echo "  - Add public key to GitHub: https://github.com/settings/keys"
        echo ""
    else
        # HTTPS
        remote_url="https://github.com/${github_user}/${repo_name}.git"
        echo ""
        echo -e "${YELLOW}Using HTTPS authentication${NC}"
        echo "You'll need a Personal Access Token:"
        echo "  1. Go to: https://github.com/settings/tokens"
        echo "  2. Click 'Generate new token (classic)'"
        echo "  3. Select scopes: repo (full control of private repositories)"
        echo "  4. Copy the token"
        echo ""
        echo "When git asks for password, use your token instead"
        echo ""
    fi
    
    git remote add origin "$remote_url"
    echo -e "${GREEN}✓ Remote configured: $remote_url${NC}"
fi
echo ""

# Step 5: Summary
echo -e "${YELLOW}[5/5] Summary${NC}"
echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Create a repository on GitHub (https://github.com/new)"
echo "  2. Run: bash push_to_github.sh"
echo ""
echo "Git configuration:"
git config --list | grep "user\|remote"
echo ""
