BattleshipSolver correction package for GitHub main branch as reviewed 2026-10-31.
Changes are NOT uploaded or committed to GitHub. Applying this ZIP updates only
your local checkout and restores intended filenames of the expanded tests.

From the repository root, first run `git status --short` and back up any
uncommitted edits. Download this ZIP to your computer and use its real path:

  unzip -o "/mnt/c/Users/YOUR_WINDOWS_USER/Downloads/battleship_correctness_repo_aligned.zip" -d .
  python -m pytest -rx -q

The source changes fix validation, no-touch halo logic, duplicate ship sinks,
and invalid-board recommendations. The Monte Carlo bias test remains expected
failure, by design, until the sampling method is rewritten.
