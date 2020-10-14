import github

# Fetches the date of the last commit to the master branch to use as version date
def get_version_date():
    g=github.Github() # instantiates github api class with no credentials
    try:
        repo = g.get_repo("AIforGoodSimulator/model-server")
        commit = repo.get_commit(sha="master") # gets recent comment to branch
        versionDate = commit.commit.author.date # extracts the commit date from the commit info
    except: # unauthenticated with github account so only 5000 requests per hour
        versionDate= "Unavailiable: Too many requests"
    return(versionDate)

