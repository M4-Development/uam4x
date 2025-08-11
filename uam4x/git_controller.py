import pygit2
import os
# from uam4x.framework import Study

AUTHOR = pygit2.Signature("DEFAULT AUTHOR", 'fake@email')
COMMITTER = pygit2.Signature("DEFAULT COMMITTER", 'fake@email')


class GitController():
    def __init__(self, directory) -> None:
        self.repo = None
        self.dir = directory
        self.initialize_study_repo()

    def initialize_study_repo(self):
        if not os.path.isdir(self.dir):
            raise NotADirectoryError("study_dir is not a directory")
        assert os.path.isdir(os.path.join(self.dir, ".git")), "There must be an initial git repository"
        self.repo = pygit2.init_repository(self.dir)

    @property
    def active_branch_name(self):
        return self.repo.head.shorthand

    def commit_data_json(self, file_path, commit_message):
        rel_path = os.path.relpath(file_path,self.dir).replace('\\', r'/')
        parents = [self.repo.head.target]
        index = self.repo.index
        index.add(rel_path)
        index.write()
        tree = index.write_tree()
        self.repo.create_commit("HEAD", AUTHOR, COMMITTER, commit_message, tree, parents)

    def get_all_branches(self):
        return list(self.repo.branches)

    def make_branch(self, branch_name, active, commit):
        if commit:
            raise NotImplementedError("cant branch specific commit")
        else:
            branch_commit = self.repo[self.repo.head.target]
        self.repo.branches.local.create(branch_name, branch_commit)
        self.repo.checkout(self.repo.branches[branch_name].name)
        if active:
            self.repo.checkout(self.repo.branches[branch_name].name)

    def switch_branch(self, branch_name):
        self.repo.checkout(self.repo.branches[branch_name].name)

    def delete_branch(self, branch_name):
        assert not branch_name == 'baseline', "can't delete baseline branch"
        self.repo.branches.delete(self.repo.branches[branch_name].name)
        # revert to master if active branch is deleted

    def rename_branch(self, old_name, new_name):
        assert not old_name == 'baseline', "can't rename baseline branch"
        self.repo.branches[old_name].rename(new_name)

    def rebase_branch_onto_baseline(self, branch_name):
        self.repo.branches[branch_name].reb
