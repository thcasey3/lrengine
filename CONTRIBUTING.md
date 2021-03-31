# Contributing to lrengine 
Please feel free to contribute by:

- Reporting a bug
- Proposing new features
- Improving documentation
- Adding a test case
- Submitting a fix

## Bug reports and enhancement requests
I use GitHub issues to track public bugs and enhancement requests:
[opening a new issue](https://github.com/thcasey3/lrengine/issues/new).

## Working with the code

### Git and Github

I use git for version control, and github to host the lrengine code
base.

Some resources for learning Git:
- [Atlassian Git tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control)
- [Pandas's contribution guidance](https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#version-control-git-and-github)

Some resources for learning GitHub:
- [Getting started with GitHub](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github)

### Forking and cloning

After [creating your free github account](https://github.com/join), you
can go to the [lrengine project page](https://github.com/thcasey3/lrengine)
and click the **Fork** button to creating a copy of the project to your
own Github account.

From now on, type commands in terminal if you are using MacOS or
Linux, or in cmd prompt if you are using windows (For Windows 10, make
sure you change to a directory with write permission).

After forking,  copy the project from Github to your local
machine by
```
git clone https://github.com/<your-github-user-name>/lrengine lrengine-yourname
cd lrengine-yourname
git remote add upstream https://github.com/thcasey3/lrengine.git
```

The `master` branch only hosts the latest release, make sure you are on
the `development` branch for the latest production-ready development. Type
```
git checkout development
git pull upstream development
```

### Creating a development environment

If you are making documentation changes only, you can skip this section.

To test code changes, first set up all the dependencies.

Using a virtual environment is the safest method. You can use conda by doing the
following:

1. Install either
   [Anaconda](https://www.anaconda.com/products/individual) or
   [miniconda](https://docs.conda.io/en/latest/miniconda.html)
   
2. Make sure conda is up-to-date
   ```
   conda update conda
   ```
3. Create an environment with Python>=3.6
   ```
   conda create --name environment_name python=3.8
   conda activate environment_name
   ```
4. Install dependencies. First, use `git status` to make sure you are on the
   `development` branch, then type
   ```
   python -m pip install -r requirements.txt
   ```

### Branching, commiting and pushing
When creating a new branch, make sure the `development` is up to
date with lrengine project. You can do so by
```
git checkout development
git pull upstream development
```

Branch from `development` when making any change.
```
git branch yourname-gh-##
git checkout yourname-gh-##
```
Use `yourname-gh-##` as the branch name, where
`##` is the corresponding issue or pull request number.

After making changes or adding features, modify an existing unittest or add at least one new unittest to ensure you are not breaking the code.
```
python -m pytest
```

Run the following to check syntax error and correct formats
```
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
python -m black .
```

Commit your changes following
[pandas: committing your code](https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#committing-your-code)

Push to your Github repository by
```
git push -u origin yourname-gh-##
```

### Pull requests
After reviewing your changes, you can file a pull request for the
maintainer of lrengine to review and approve. See
[Github creating-a-pull-request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)

Make sure you are requesting to merge to `lrengine/development` from
`your-github-username/yourname-gh-##`.

Your changes will trigger multiple automatic checks to ensure it
won't break the package. Please always add a unittest when a feature is added, or adjust the current unittest accordingly if a feature is changed. I will accept or provide revising comments.

## Contributing to documentation
Same general guidlines as above.

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

