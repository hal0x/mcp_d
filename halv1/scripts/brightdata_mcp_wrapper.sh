#!/bin/bash

# Bright Data MCP Wrapper Script
export API_TOKEN="${BRIGHTDATA_API_TOKEN}"
export NODE_ENV="production"

cd /Users/hal/projects/halv1
exec /opt/homebrew/bin/npx @brightdata/mcp



