#!/bin/bash

# Conversation Agent - GitHub Push Script
# Pushes the project to GitHub repository

set -e  # Exit on error

echo "=========================================="
echo "Conversation Agent - GitHub Push"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_DIR")

# Step 1: Check git status
echo -e "${YELLOW}[1/6] Checking git status...${NC}"
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    echo "Run 'bash setup_github.sh' first"
    exit 1
fi

git_status=$(git status --porcelain)
if [ -z "$git_status" ]; then
    echo -e "${GREEN}✓ Working directory clean${NC}"
else
    echo -e "${YELLOW}Uncommitted changes found:${NC}"
    git status --short
fi
echo ""

# ---- .gitignore (create if missing) ----
if [ ! -f ".gitignore" ]; then
  cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env

# Node
node_modules/
dist/
.vite/

# Data outputs (keep directory)
data/*.csv
!data/.gitkeep

# OS / editor
.DS_Store

slm-eval-ide-prompt.md

scripts/

EOF
fi

# Step 2: Configure .gitignore
echo -e "${YELLOW}[2/6] Verifying .gitignore...${NC}"
if [ -f ".gitignore" ]; then
    echo -e "${GREEN}✓ .gitignore exists${NC}"
else
    echo -e "${RED}Warning: .gitignore not found${NC}"
fi
echo ""

# Step 3: Stage all files
echo -e "${YELLOW}[3/6] Staging files...${NC}"
echo "Files to be committed:"
echo ""
git add .
git diff --cached --name-status | head -20
echo ""
file_count=$(git diff --cached --name-only | wc -l)
echo -e "${GREEN}✓ Staged $file_count files${NC}"
echo ""

# Step 4: Commit changes
echo -e "${YELLOW}[4/6] Creating initial commit...${NC}"
if git rev-parse HEAD > /dev/null 2>&1; then
    echo -e "${YELLOW}Repository already has commits${NC}"
    read -p "Create a new commit? (y/n): " create_commit
    if [ "$create_commit" = "y" ]; then
        read -p "Enter commit message (default: 'Update RAG system'): " commit_msg
        commit_msg=${commit_msg:-"Update RAG system"}
        git commit -m "$commit_msg" || echo -e "${YELLOW}No changes to commit${NC}"
    fi
else
    read -p "Enter commit message (default: 'Initial commit: RAG Assistant'): " commit_msg
    commit_msg=${commit_msg:-"Initial commit: RAG Assistant"}
    git commit -m "$commit_msg"
    echo -e "${GREEN}✓ Initial commit created${NC}"
fi
echo ""

# Step 5: Check remote
echo -e "${YELLOW}[5/6] Checking remote configuration...${NC}"
if git remote get-url origin &> /dev/null; then
    remote_url=$(git remote get-url origin)
    echo -e "${GREEN}✓ Remote configured: $remote_url${NC}"
else
    echo -e "${RED}Error: No remote configured${NC}"
    echo "Run 'bash setup_github.sh' first"
    exit 1
fi
echo ""

# Step 6: Push to GitHub
echo -e "${YELLOW}[6/6] Pushing to GitHub...${NC}"
echo ""
echo -e "${BLUE}Note: You may be prompted for credentials${NC}"
echo ""

# Check current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Pushing '$current_branch' branch to remote..."
echo ""

if git push -u origin "$current_branch" 2>&1; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ Successfully pushed to GitHub!"
    echo "==========================================${NC}"
    echo ""
    echo "Repository URL: $remote_url"
    echo "Branch: $current_branch"
    echo ""
    echo "Next steps:"
    echo "  1. Visit your repository: ${remote_url%.git}"
    echo "  2. Add collaborators in Settings"
    echo "  3. Configure CI/CD if needed"
    echo ""
else
    echo ""
    echo -e "${RED}Error: Failed to push to GitHub${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Verify remote URL: git remote -v"
    echo "  2. Check authentication:"
    echo "     - HTTPS: Use Personal Access Token as password"
    echo "     - SSH: Ensure SSH keys are set up"
    echo "  3. Verify repository exists on GitHub"
    echo ""
    exit 1
fi
