# About Git commands

Nir Zadok

April 18, 2021

# Contents

1. Introduction
2. Using Google Drive in Google Colab
3. Using Git
   1. Initializing
   2. On basis commands
      1. Used all-the-time commands
   3. About branches
   4. The workflow

# Introduction

In this markdown file I will introduce useful Git commands and the workflow of Google Colab vs. Github.

# Using Google Drive in Google Colab

In order to use get access to Google Drive trough Google Colab, one has to
use the following commands:

``` python
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
```

Then one has to change folder to our working directory:

```python
%cd /content/gdrive/My Drive/Colab Notebooks/{folder_name}/
```

 Now one can see the files inside the directory and also the path by using:

```python
!ls
!pwd
```

# Using Git

Now we will start using Git in order to connect our Google Colab and Google
Drive with Github.

## Initializing

First, we will go to our Drive folder and check if there exists a .git folder:

![image-20210419080502291](C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20210419080502291.png)

If there is, delete it by using:

```python
rm -rf .git
```

Then, we will have to initialize empty Git repository by using:

```python
!git init
```

Next, we will define our remote repository on Github:

```python
!git remote add origin https://{Github_username}:{Github_password}
@github.com/{Github_username}/{repository_name}.git
```

If there is a problem while creating it, use the following command:

```python
!git remote rm origin
```

Now we have to config our account:

```python
!git config --global user.email "{Github_mail}"
!git config --global user.name "{Github_username}"
```

In order to get updated from the Github repository:

```python
!git fetch
```

## On-basis commands

### Used all-the-time commands

1. The status command helps us learning about our status:

   ```python
   !git status
   ```

2. In order to refresh the git, we will have to use:

   ```python
   !git reset
   ```

   If Git is worried about files unwritten, use these commands, while add
   is responsible for adding and stash is responsible for the saving:

   ```python
   !git add --all
   !git commit -m "save the unwritten files"
   !git stash
   ```

   In order to get a list of the stashed files, one can run:

   ```python
   !git stash list
   ```

   Then one is able to load a stash by using:

   ```python
   !git stash apply {stashed_file}
   ```

   In order to get clone of the remote repository, one can run:

   ```python
   !git clone {repo_link}
   ```

## About branches

1. Check in which branch you are by using:

   ```python
   !git branch
   ```

   For example, here we are at new-feature branch:
   ![image-20210419081206106](C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20210419081206106.png)

   We can see it also by using:

   ```python
   !git status
   ```

   ![image-20210419081223140](C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20210419081223140.png)

   2. We can change to the desired branch by using:

      ```python
      !git checkout my_branch
      ```

   3. We can also create new branch by using:

      ```python
      !git checkout -b my_branch
      ```

   4. And also create a new branch like the active branch (say we are inside
      our branch and we want a copy from it):

      ```python
      !git branch new-branch
      ```

      ```python
      !git checkout new-branch
      ```

## The workflow

1. We start inside our working branch

2. Next we will use the commad:

   ```python
   !ls
   ```

   in order to check what files we have in this Google Drive directory,
   and also use:

   ```python
   !git status
   ```

   to check what our status is.

3. We will begin by reset our commit:

   ```python
   !git reset
   ```

   And also create a new branch like the active branch (say we are inside
   our branch and we want a copy from it):

   ```python
   !git branch new-branch
   ```

   ```python
   !git checkout new-branch
   ```

4. Now we can add a file to our working branch by using:

   ```python
   !git add {file_name}
   ```

   Then we will have to create a commit, by using:

   ```python
   !git commit -m "Push files into a branch"
   ```

   ```python
   !git push origin {branch_name}
   ```

   We will go to the branch we want to merge into it:

   ```python
   !git checkout {desired_branch}
   ```

   Now, will reset first:

   ```python
   !git reset --hard {merged_branch}
   ```

   We will add our commit:

   ```python
   !git commit -m "Merge the branch into another branch"
   ```

   and pull to get all our data before merging:

   ```python
   !git pull origin {the_branch_we_merge_into}
   ```

   Or the same if to get the data from the remote repository by using:

   ```python
   !git fetch
   ```

   and then to use merge:

   ```python
   !git merge {merged_branch}
   ```

   And finally we will commit and push it:

   ```python
   !git commit -m "merge it"
   !git push origin main
   ```

   Then one can delete the branch from the remote repository and also
   from the local repository:

   ```python
   !git branch -d my_branch
   ```

   