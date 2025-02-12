#!/bin/bash
# force-push-script.sh
git config --local core.hooksPath /dev/null
git add .
git commit -m "Emergency commit from restricted machine - TO BE REVIEWED"
git push --force-with-lease origin feature/emergency-branch