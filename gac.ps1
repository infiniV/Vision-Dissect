#!/usr/bin/env pwsh
# Git Add, Commit with AI, and Push script

# Check if CEREBRAS_API_KEY is set
if (-not $env:CEREBRAS_API_KEY) {
    Write-Host "Error: CEREBRAS_API_KEY environment variable is not set" -ForegroundColor Red
    Write-Host "Get your free API key at: https://cloud.cerebras.ai" -ForegroundColor Yellow
    exit 1
}

# Stage all changes
Write-Host "Staging changes..." -ForegroundColor Cyan
git add -A

# Get git status summary
$gitStatus = git status --short
$gitDiff = git diff --cached --stat

if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Host "No changes to commit" -ForegroundColor Yellow
    exit 0
}

# Count file types and changes
$fileCount = ($gitStatus | Measure-Object).Count
$statusLines = $gitStatus -split "`n" | Select-Object -First 10
$statusSummary = if ($fileCount -gt 10) { 
    "$($statusLines -join "`n")`n... and $($fileCount - 10) more files" 
} else { 
    $gitStatus 
}

# Prepare the prompt - simplified to avoid huge payloads
$userPrompt = "Generate a proper git commit message for these changes:`n`nFiles changed: $fileCount`n`n$statusSummary`n`nStats: $gitDiff`n`nFormat:`n- First line: concise summary (max 72 chars)`n- Blank line`n- Body: bullet points explaining what changed and why (2-4 lines)`n`nReturn ONLY the commit message, no quotes or markdown."

# Create JSON payload using PowerShell objects
$bodyObject = @{
    model = "llama-3.3-70b"
    messages = @(
        @{
            role = "system"
            content = "You are a git commit message generator. Generate proper multi-line conventional commit messages with a summary line followed by a blank line and a detailed body. Return only the commit message without quotes, markdown, or formatting."
        }
        @{
            role = "user"
            content = $userPrompt
        }
    )
    max_completion_tokens = 250
    temperature = 0.6
    top_p = 0.95
    stream = $false
}

$body = $bodyObject | ConvertTo-Json -Depth 10

Write-Host "Generating commit message..." -ForegroundColor Cyan

try {
    # Call Cerebras API
    $response = Invoke-RestMethod -Uri "https://api.cerebras.ai/v1/chat/completions" `
        -Method Post `
        -Headers @{
            "Content-Type" = "application/json"
            "Authorization" = "Bearer $env:CEREBRAS_API_KEY"
        } `
        -Body $body

    # Extract commit message
    $commitMsg = $response.choices[0].message.content.Trim()
    
    # Remove any markdown formatting or quotes
    $commitMsg = $commitMsg -replace '^\*\*', '' -replace '\*\*$', '' -replace '^"', '' -replace '"$', '' -replace '^```', '' -replace '```$', ''
    
    # Clean up any leading/trailing whitespace while preserving internal structure
    $commitMsg = $commitMsg.Trim()
    
    Write-Host "`nGenerated commit message:" -ForegroundColor Green
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Write-Host $commitMsg -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
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
