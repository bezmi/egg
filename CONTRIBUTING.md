# Contributing

Thanks for your interest in contributing to Egg! This document
explains how to contribute and the terms under which contributions are accepted.

## License

This project is licensed under the terms in [LICENSE.md](./LICENSE.md). Please
read it before contributing so you understand how the project is distributed.

Note that this is a **source-available** license, not an OSI-approved open
source license — free for personal, hobby, and academic use. You may run, copy,
modify, and share the software and your changes, **provided all use stays
noncommercial** and you pass along this license and any required notices. To
protect community contributions, **each tagged release's artifacts convert to GNU
AGPL-3.0 four years after its release date**, via the PolyForm Countdown notice
attached to that release.

Contributing requires agreeing to the Contributor License Agreement (CLA) at the
end of this document, which—among other things—allows the maintainer to offer
the project (including your contributions) under separate commercial terms. See
[Signing the CLA](#signing-the-cla) below for how to record your agreement.

### Signing the CLA

Before your contribution can be merged, you must agree to the [Contributor
License Agreement](#contributor-license-agreement-cla) below. To do so, leave a
comment on your pull request stating:

> I have read and agree to the Contributor License Agreement.

Please include this confirmation on your first pull request. Your agreement
applies to that contribution and any future contributions you make to the
project.

**Corporate contributions:** if you are contributing on behalf of an employer,
please contact <s.imran@tuta.io> to arrange a Corporate CLA before submitting.

## How to contribute

1. **Open an issue first** for anything non-trivial, so we can discuss the
   approach before you invest time in it.
2. **Fork** the repository and create a feature branch off `main`.
3. Make your changes, including tests and documentation where applicable.
4. Ensure the test suite and any linters pass locally.
5. **Open a pull request** against `main` and fill out the PR template.
6. A maintainer will review. Address feedback by pushing additional commits to
   the same branch.

### Commit messages: Conventional Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
Every commit message must follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Allowed types:**

- `feat:` — a new feature
- `fix:` — a bug fix
- `docs:` — documentation only
- `style:` — formatting, no code-meaning change
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `perf:` — a performance improvement
- `test:` — adding or correcting tests
- `build:` — build system or dependency changes
- `ci:` — CI configuration changes
- `chore:` — other changes that don't modify src or test files

**Examples:**

```
feat(parser): add support for nested expressions
fix(api): handle null response from upstream service
docs: clarify installation steps for Windows
```

**Breaking changes:** append `!` after the type/scope and explain in a
`BREAKING CHANGE:` footer, e.g.:

```
feat(api)!: drop support for legacy auth tokens

BREAKING CHANGE: the `token_v1` field has been removed; use `token` instead.
```

PRs whose commits don't follow this convention may be asked to amend or
rebase before merge.

## Contributor License Agreement (CLA)

You agree to this Agreement by stating, in a comment on your pull request, that
You have read and agree to the Contributor License Agreement (as described in
[Signing the CLA](#signing-the-cla) above). That confirmation makes this
Agreement effective for all past and future Contributions You submit to the
project. Except for the license granted herein, You reserve all right, title,
and interest in and to Your Contributions.

**1. Definitions.**
"You" (or "Your") means the copyright owner or legal entity authorized by the
copyright owner that is entering into this Agreement. "Contribution" means any
original work of authorship, including any modifications or additions to an
existing work, that is intentionally submitted by You to the maintainer of this
project (the "Maintainer") for inclusion in, or documentation of, the project.

**2. Grant of Copyright License.**
Subject to the terms of this Agreement, You hereby grant to the Maintainer and
to recipients of software distributed by the Maintainer a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable copyright license to
reproduce, prepare derivative works of, publicly display, publicly perform,
sublicense, and distribute Your Contributions and such derivative works.

**3. Right to Relicense.**
You expressly acknowledge and agree that the Maintainer may license and
distribute Your Contributions, and any derivative works thereof, under any
license terms whatsoever, including proprietary or commercial terms, and that
the license granted in Section 2 is sublicensable for that purpose. You will
not be entitled to any compensation in connection with such licensing or
distribution.

**4. Grant of Patent License.**
Subject to the terms of this Agreement, You hereby grant to the Maintainer and
to recipients of software distributed by the Maintainer a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this
section) patent license to make, have made, use, offer to sell, sell, import,
and otherwise transfer Your Contribution, where such license applies only to
those patent claims licensable by You that are necessarily infringed by Your
Contribution alone or by combination of Your Contribution with the project to
which it was submitted. If any entity institutes patent litigation against You
or any other entity alleging that Your Contribution, or the project to which You
contributed, constitutes direct or contributory patent infringement, then any
patent licenses granted to that entity under this Agreement for that
Contribution or project shall terminate as of the date such litigation is filed.

**5. Representations.**
You represent that You are legally entitled to grant the above licenses. If Your
employer(s) has rights to intellectual property that You create that includes
Your Contributions, You represent that You have received permission to make
Contributions on behalf of that employer, that Your employer has waived such
rights for Your Contributions, or that Your employer has executed a separate
Corporate CLA with the Maintainer.

**6. Original Work.**
You represent that each of Your Contributions is Your original creation. You
represent that Your Contribution submissions include complete details of any
third-party license or other restriction (including, but not limited to,
related patents and trademarks) of which You are personally aware and which are
associated with any part of Your Contributions.

**7. No Obligation.**
You understand that the decision to include Your Contribution in any project or
source repository is entirely that of the Maintainer, and this Agreement does
not guarantee that the Contributions will be included in any product.

**8. Disclaimer.**
Unless required by applicable law or agreed to in writing, You provide Your
Contributions on an "AS IS" basis, without warranties or conditions of any kind,
either express or implied, including, without limitation, any warranties or
conditions of title, non-infringement, merchantability, or fitness for a
particular purpose.
