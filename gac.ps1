#!/usr/bin/env pwsh
# Git Add, Commit with AI, and Push script

# Check if GROQ_API_KEY is set
if (-not $env:GROQ_API_KEY) {
    Write-Host "Error: GROQ_API_KEY environment variable is not set" -ForegroundColor Red
    exit 1
}

# Stage all changes
Write-Host "Staging changes..." -ForegroundColor Cyan
git add -A

# Get git status summary
$gitStatus = git status --short
$gitStats = git diff --cached --stat

if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Host "No changes to commit" -ForegroundColor Yellow
    exit 0
}

# Prepare the prompt for Groq with just the summary
$prompt = "Generate a simple, concise git commit message (one line, max 10 words, no quotes or formatting) for these file changes:`n`n$gitStatus`n`nStats: $gitStats"

# Create JSON payload
$messages = @(
    @{
        role = "user"
        content = $prompt
    }
) | ConvertTo-Json -Depth 10

$body = @{
    messages = @(
        @{
            role = "user"
            content = $prompt
        }
    )
    model = "moonshotai/kimi-k2-instruct-0905"
    temperature = 0.6
    max_completion_tokens = 100
    top_p = 1
    stream = $false
} | ConvertTo-Json -Depth 10

Write-Host "Generating commit message..." -ForegroundColor Cyan

try {
    # Call Groq API
    $response = Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/chat/completions" `
        -Method Post `
        -Headers @{
            "Content-Type" = "application/json"
            "Authorization" = "Bearer $env:GROQ_API_KEY"
        } `
        -Body $body

    # Extract commit message
    $commitMsg = $response.choices[0].message.content.Trim()
    
    # Remove any markdown formatting or quotes
    $commitMsg = $commitMsg -replace '^\*\*', '' -replace '\*\*$', '' -replace '^"', '' -replace '"$', '' -replace '^`', '' -replace '`$', ''
    
    Write-Host "Commit message: $commitMsg" -ForegroundColor Green
    
    # Commit with the generated message
    git commit -m $commitMsg
    
    # Push to remote
    Write-Host "Pushing to remote..." -ForegroundColor Cyan
    git push
    
    Write-Host "Done!" -ForegroundColor Green
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Failed to generate commit message. Aborting." -ForegroundColor Red
    exit 1
}
