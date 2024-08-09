#!/usr/bin/env python3

import re,os,sys,copy,subprocess,shutil,glob,argparse
try:
    import git
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True
except ImportError:
    print("\nFatal error: modules not available.")
    print("On NCI, do the following and try again:")
    print("   module use /g/data/vk83/modules && module load payu/1.1.4\n")
    raise

DIR_MANAGER     = os.getcwd()
DIR_UTILS       = os.path.join(DIR_MANAGER,"tools/om3-utils")
BRANCH_PERTURB  = "perturb"

try:
    # currently imported from a fork: https://github.com/minghangli-uni/om3-utils,
    # will update the tool when it is merged to COSIMA/om3-utils
    #utils_name = "om3-utils"
    #utils_branch_name = "main"
    #utils_template = "https://github.com/minghangli-uni/om3-utils"
    #command = f"payu clone -B {utils_branch_name} -b {utils_branch_name} {utils_template} {self.base_path}"
    #subprocess.run(command, shell=True, check=False)
    sys.path.append(DIR_UTILS)
    repo = git.Repo(DIR_UTILS)
    repo.git.checkout("main")
    from om3utils import MOM6InputParser
    from om3utils.nuopc_config import read_nuopc_config, write_nuopc_config

except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Failed to import om3utils. Check the sys.path to ensure the module existing in the dir.")
    raise
# ===============================================

def update_MOM6_params_override(param_dict_change, commt_dict_change):
    # needs prepend `#override `
    override_param_dict_change = {f"#override {k}": v for k,v in param_dict_change.items()}
    override_commt_dict_change = {f"#override {k}": v for k,v in commt_dict_change.items()}
    return override_param_dict_change, override_commt_dict_change

class Expts_manager(object):
    def __init__(self, dir_manager=DIR_MANAGER):
        self.dir_manager = dir_manager
        self.template_path = None
        self.param_dict_change_list = []
        self.param_dict_change_full_list = []
        self.commt_dict_change = {}
        self.startfrom = None
        self.expt_yaml = None
        self.base_branch_name = None
        self.base_path = None
        self.test_path = None
        self.expt_names = None
        
    def setup_ctrl_expt(self,yamlfile):
        self.expt_yaml = self._read_ryaml(yamlfile)  # load Expts_manager.yaml 
        template_url = self.expt_yaml["template_url"]  # template git url
        template_commit = self.expt_yaml["template_commit"]  # specific git commit 
        self.base_dir_name = self.expt_yaml["base_dir_name"]  # user-defined directory for a baseline control experiment
        self.base_branch_name = self.expt_yaml["base_branch_name"]  # user-defined branch name for the control experiment
        self.test_path = self.expt_yaml["test_path"]  # user-defined path for the test runs, including the control experiment and multiple parameter-tunning experiments
        
        # detect parameter changes in `Expts_manager.yaml`
        namelists = self.expt_yaml["namelists"]
        if namelists is not None:
            if "ice/cice_in.nml" in namelists and "MOM_list" not in namelists:
                if namelists["ice/cice_in.nml"] is None:
                    raise ValueError("None parameter is tuned for CICE!")
                else:
                    pass
            elif "ice/cice_in.nml" in namelists and "MOM_list" in namelists:
                raise ValueError("namelists must contain either 'ice/cice_in.nml' or 'MOM_list'")
            elif "ice/cice_in.nml" not in namelists and "MOM_list" in namelists:
                if namelists["MOM_list"] is None:
                    raise ValueError("None parameter is tuned for MOM6!")
                else:
                    name_dict = namelists["MOM_list"]
        else:
            raise ValueError("namelists can't be None!")

        # if exist expt runs, then create new paths
        if os.path.exists(self.test_path):
            print(f"test directory {self.test_path} already exists!")
        else:
            os.makedirs(self.test_path)
            print(f"test directory {self.test_path} is created!")

        # create the path for the ctrl base run
        self.base_path = os.path.join(self.test_path,self.base_dir_name)

        # create ctrl branch from a designated repo defined in `Expts_manager.yaml`
        if os.path.exists(self.base_path):
            print(f"Base path is already created located at {self.base_path}")
        else:
            print(f"Cloning the template from {template_url} to {self.base_path}")
            command = f"payu clone {template_url} {self.base_path}"
            subprocess.run(command, shell=True, check=False)
            templaterepo = git.Repo(self.base_path)
            print(f"Checkout commit {template_commit} - corresponding to repo's branch: {templaterepo.active_branch.name};")
            print(f"and create a new branch called {self.base_branch_name} for the control run!")
            templaterepo.git.checkout('-b', self.base_branch_name, template_commit)  # checkout the new branch from the specific template commit
            
            
        # Update nuopc.runconfig for the ctrl run
        nuopc_input = self.expt_yaml["nuopc_runconfig"]
        if nuopc_input is not None:
            nuopc_file_path = os.path.join(self.base_path,"nuopc.runconfig")
            nuopc_runconfig = read_nuopc_config(nuopc_file_path)
            self._update_configs(nuopc_runconfig,nuopc_input)
            write_nuopc_config(nuopc_runconfig, nuopc_file_path)

        # Update config.yaml for the ctrl run
        config_yaml_input = self.expt_yaml["config_yaml"]
        if config_yaml_input is not None:
            config_yaml_file = os.path.join(self.base_path,"config.yaml")
            config_yaml = self._read_ryaml(config_yaml_file)
            self._update_configs(config_yaml,config_yaml_input)
            self._write_ryaml(config_yaml_file,config_yaml)

        # Update coupling timestep through nuopc.runseq for the ctrl run
        cpl_dt_input = self.expt_yaml["cpl_dt"]
        if cpl_dt_input is not None:
            nuopc_runseq_file = os.path.join(self.base_path,"nuopc.runseq")
            self._update_cpl_dt_nuopc_seq(nuopc_runseq_file,cpl_dt_input)

        # Payu setup && sweep to ensure the changes correctly && remove the `work` directory for the ctrl run
        command = f"cd {self.base_path} && payu setup && payu sweep"
        subprocess.run(command, shell=True, check=False)

        # check file changes and commits if so, otherwise, no commits for the ctrl run.
        repo = git.Repo(self.base_path)
        print(f"Current base branch is: {repo.active_branch.name}")
        changed_files = self._get_changed_files(repo)
        if changed_files:
            print(f"Configure `{self.base_branch_name}` branch in preparation for expt runs!")
            repo.index.add(changed_files)
            commit_message = f"Configure `{self.base_branch_name}` branch by `{yamlfile}` in preparation for expt runs!"
            repo.index.commit(commit_message)
        else:
            print(f"Nothing changed, hence no further commits to the {self.base_path} repo!")

        # if MOM6 change, parse MOM_input parameters, values and comments
        if name_dict:
            MOM_inputParser = self._parser_mom6_input(os.path.join(self.base_path, "MOM_input"))
            param_dict = MOM_inputParser.param_dict
            commt_dict = MOM_inputParser.commt_dict
            self._update_param_dict_change(name_dict, param_dict, commt_dict)

        # define `startfrom` following `Expts_manager.yaml`
        self.startfrom = str(self.expt_yaml["startfrom"]).strip().lower().zfill(3)

    def _parser_mom6_input(self, path_file):
        """ parse MOM6 input file """
        mom6parser = MOM6InputParser.MOM6InputParser()
        mom6parser.read_input(path_file)
        mom6parser.parse_lines()
        return mom6parser

    def _update_configs(self,base,change):
        """ recursively update nuopc_runconfig and config.yaml entries """
        for k,v in change.items():
            if isinstance(v,dict) and k in base:
                self._update_configs(base[k],v)
            else:
                base[k] = v

    def _update_cpl_dt_nuopc_seq(self,seq_path,update_cpl_dt):
        """ update coupling timestep through nuopc.runseq"""
        with open(seq_path,"r") as f:
            lines = f.readlines()
        pattern = re.compile(r"@(\S*)")
        update_lines = []
        for l in lines:
            matches = pattern.findall(l)
            if matches:
                update_line = re.sub(r"@(\S+)", f"@{update_cpl_dt}", l)
                update_lines.append(update_line)
            else:
                update_lines.append(l)
        with open(seq_path,"w") as f:
            f.writelines(update_lines)

    def _update_param_dict_change(self, name_dict, param_dict, commt_dict):
        """ load parameters, values and associated comments from the ctrl expt"""
        num_expts = len(next(iter(name_dict.values())))  # number of expts
        keys = name_dict.keys()
        param_dict_change_full_list = []  # A list includes full parameter input dicts for all tests
        param_dict_change_list      = []  # A list includes only changed parameter input dicts for all tests
        self.commt_dict_change      = {key: commt_dict.get(key,"") for key in keys}

        for i in range(num_expts):
            tmp_param_dict_full = copy.deepcopy(param_dict)
            param_dict_change   = {key: name_dict[key][i] for key in keys}
            if any(param_dict[key] != param_dict_change[key] for key in keys):
                for key in keys:
                    tmp_param_dict_full[key] = param_dict_change[key]
                param_dict_change_full_list.append(tmp_param_dict_full)
                param_dict_change_list.append(param_dict_change)

        self.param_dict_change_list = param_dict_change_list
        self.param_dict_change_full_list = param_dict_change_full_list
        
    def manage_expts(self):
        """ setup expts, and run expts"""
        for i in range(len(self.param_dict_change_list)):
            # for each experiment
            expts_input = self.expt_yaml["expt_names"]  # user-defined folder names for parameter-tunning experiments.
            if expts_input is None:
                nuopc_file_path = os.path.join(self.base_path,"nuopc.runconfig")
                expt_name = "_".join([f"{k}_{v}" for k,v in self.param_dict_change_list[i].items()])
            else:
                expt_name = expts_input[i]
            rel_path = os.path.join(self.test_path,expt_name)
            expt_path = os.path.join(self.dir_manager,rel_path)
            print(f"\n {expt_path}")
            if os.path.exists(expt_path):
                print("-- not creating ", rel_path, " - already exists!")
            else:
                print("clone template - payu clone!")
                command = f"payu clone -B {self.base_branch_name} -b {BRANCH_PERTURB} {self.base_path} {expt_path}" # automatically leave a commit with expt uuid
                subprocess.run(command, shell=True, check=True)

                # apply changes and write them to `MOM_override`
                MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, "MOM_override"))
                override_param_dict_change, override_commt_dict_change = update_MOM6_params_override(self.param_dict_change_list[i],self.commt_dict_change)
                MOM6_or_parser.param_dict = override_param_dict_change
                MOM6_or_parser.commt_dict = override_commt_dict_change
                MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))

                # Update config.yaml 
                config_path = os.path.join(expt_path,"config.yaml")
                config_data = self._read_ryaml(config_path)
                config_data["jobname"] = "_".join([self.base_branch_name,expt_name])
                self._write_ryaml(config_path,config_data)

                # Update metadata.yaml
                metadata_path = os.path.join(expt_path, "metadata.yaml")
                metadata = self._read_ryaml(metadata_path)
                self._update_metadata_description(metadata)
                self._remove_metadata_comments("description", metadata)
                keywords = self._extract_metadata_keywords(self.param_dict_change_list[i])
                metadata["keywords"] = f"{self.base_dir_name}, {BRANCH_PERTURB}, {keywords}"
                self._remove_metadata_comments("keywords", metadata)
                self._write_ryaml(metadata_path, metadata)

                # replace metadata in archive/
                shutil.copy(metadata_path,os.path.join(expt_path,"archive"))

                # commit the above changes for expt runs
                exptrepo = git.Repo(expt_path)
                changed_files = self._get_changed_files(exptrepo)
                untracked_files = self._get_untracked_files(exptrepo)
                files_to_stages = set(changed_files+untracked_files)
                if files_to_stages:  # commit changes for each expt run
                    print(f"files need to be staged: {files_to_stages}")
                    exptrepo.index.add(files_to_stages)
                    commit_message = f"Payu clone from the base: {self.base_path};\nCommitted files/directories are: {', '.join(f'{i}' for i in files_to_stages)}"
                    exptrepo.index.commit(commit_message)
                else:
                    print("No files are required to be committed!")

            # start runs
            if self.startfrom == "rest":
                print("start perturbation from rest!")
            else:  # WORKING ON
                restart_path = os.path.join(expt_path,"archive","restart",self.startfrom)

            if self.expt_yaml["nruns"] > 0:
                doneruns = len(glob.glob(os.path.join(expt_path,"archive","output[0-9][0-9][0-9]*")))
                newruns = self.expt_yaml["nruns"] - doneruns
                if newruns > 0:
                    command = f"cd {expt_path} && payu run -n {newruns}"
                    subprocess.run(command, check=False, shell=True)
                else:
                    print(f"{expt_path} has already completed {doneruns} runs! Hence stop without running!")

    def _get_untracked_files(self,repo):
        """ get untracked git files """
        return repo.untracked_files

    def _get_changed_files(self,repo):
        """ get changed git files """
        return [file.a_path for file in repo.index.diff(None)]   

    def _read_ryaml(self, yaml_path):
        """ Read yaml file and preserve comments"""
        with open(yaml_path, "r") as f:
            return ryaml.load(f)

    def _write_ryaml(self,yaml_path,data):
        """ Write yaml file and preserve comments"""
        with open(yaml_path, "w") as f:
            ryaml.dump(data,f)

    def _update_metadata_description(self, metadata):
        """Update metadata description with experiment details."""
        desc = metadata["description"]
        if desc is None:
            desc = ""
        desc += (f"\nNOTE: this is a perturbation experiment, but the description above is for the control run."
                f"\nThis perturbation experiment is based on the control run {self.base_path} from {self.base_branch_name}")
        if self.startfrom == "rest":
            desc += "\nbut with condition of rest."
        metadata["description"] = desc

    def _remove_metadata_comments(self, key, metadata):
        """Remove comments after the key in metadata."""
        if key in metadata:
            metadata.ca.items[key] = [None, None, None, None]

    def _extract_metadata_keywords(self, param_change_dict):
        """Extract keywords from parameter change dictionary."""
        keywords = ", ".join(param_change_dict.keys())
        return keywords

    def main(self):
        """ Main() function for the program """
        parser = argparse.ArgumentParser(description="Manage ACCESS-OM3 experiments.\
                 Latest version and help: https://github.com/minghangli-uni/Expts_manager")
        parser.add_argument("INPUT_YAML", type=str, nargs="?",default="Expts_manager.yaml",
                            help="YAML file specifying parameter values for expt runs. Default is Expts_manager.yaml")
        args = parser.parse_args()
        INPUT_YAML = vars(args)["INPUT_YAML"]

        yamlfile     = os.path.join(DIR_MANAGER,INPUT_YAML)
        self.setup_ctrl_expt(yamlfile)
        self.manage_expts()

# ============================================================================= # 
    # def manage_expts_bk(self):
    #     for i in range(len(self.param_dict_change_list)):
    #         # for each experiment
    #         expt_name = '_'.join([f"{k}_{v}" for k,v in self.param_dict_change_list[i].items()]) 
    #         rel_path  = '_'.join([EXPT_REL_PATH,expt_name])
    #         expt_path = os.path.join(self.dir_manager,rel_path)
    #         print(f"\n {expt_path}")

    #         if os.path.exists(expt_path):
    #             print("-- not creating ", rel_path, " - already exists!")
                
    #         else:
    #             print("clone template - payu clone!")
    #             command = f"payu clone -B {BRANCH_NAME_BASE} -b {BRANCH_PERTURB} {self.template_path} {expt_path}" # automatically leave a commit with expt uuid
    #             test = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                
    #             # apply changes and write them to `MOM_override`
    #             MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, "MOM_override"))
    #             override_param_dict_change, override_commt_dict_change = update_MOM6_params_override(self.param_dict_change_list[i],self.commt_dict_change)
    #             MOM6_or_parser.param_dict = override_param_dict_change
    #             MOM6_or_parser.commt_dict = override_commt_dict_change
    #             MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))
                
    #             # Update config.yaml 
    #             config_path = os.path.join(expt_path,"config.yaml")
    #             config_data = self._read_ryaml(config_path)
    #             config_data["jobname"] = "_".join([BRANCH_NAME_BASE,expt_name])
    #             self._write_ryaml(config_path,config_data)
    
    #             # Update metadata.yaml
    #             metadata_path = os.path.join(expt_path, "metadata.yaml")
    #             metadata = self._read_ryaml(metadata_path)
    #             self._update_metadata_description(metadata)
    #             self._remove_metadata_comments("description", metadata)
    #             keywords = self._extract_metadata_keywords(self.param_dict_change_list[i])
    #             metadata["keywords"] = f"{BRANCH_NAME_BASE}, {BRANCH_PERTURB}, {keywords}"
    #             self._remove_metadata_comments("keywords", metadata)
    #             self._write_ryaml(metadata_path, metadata)

    #             # replace metadata in archive/
    #             shutil.copy(metadata_path,os.path.join(expt_path,"archive"))
    
    #             repo = git.Repo(expt_path)
                
    #             changed_files = self._get_changed_files(repo)
    #             untracked_files = self._get_untracked_files(repo)
    #             files_to_stages = set(changed_files+untracked_files)
                
    #             if files_to_stages:  # commit changes for the expt runs
    #                 print(f"files need to be staged: {files_to_stages}")
    #                 repo.index.add(files_to_stages)
    #                 commit_message = f"Payu clone from the base: {self.template_path}; committed files/directories are: {', '.join(f'{i}' for i in files_to_stages)}"
    #                 repo.index.commit(commit_message)
    #             else:
    #                 print("No files are required to be committed!")

    #         if self.startfrom == "rest":
    #             print("start perturbation from rest!")
    #         else:  # WORKING ON
    #             restart_path = os.path.join(expt_path,"archive","restart",self.startfrom)

    #         if self.expt_yaml["nruns"] > 0:
    #             doneruns = len(glob.glob(os.path.join(expt_path,"archive","output[0-9][0-9][0-9]*")))
    #             newruns = self.expt_yaml["nruns"] - doneruns
    #             if newruns > 0:
    #                 command = f"cd {expt_path} && payu run -n {newruns}"
    #                 subprocess.run(command, check=False, shell=True)
    #             else:
    #                 print(f"{expt_path} has already completed {doneruns} runs! Hence stop without running!")
                    
    # def _get_previous_commit_messages(self,repo, num_commits):  # not used now
    #     messages = []
    #     commit = repo.head.commit  # get the head commit
    #     for _ in range(num_commits):
    #         messages.append(commit.message.strip())
    #         if commit.parents:
    #             commit = commit.parents[0]
    #         else:
    #             break
    #     return messages
        
    # def _rebase_and_squash(self, repo, num_commits):    # not used now
    #     previous_messages = self._get_previous_commit_messages(repo,num_commits) 
    #     combined_messages =  "\n\n".join(previous_messages)
    #     repo.git.rebase("-i",f"HEAD~{num_commits}")
    #     rebase_file_path = os.path.join(repo.git_dir,"/rebase-merge/git-rebase-todo")  # rebase instructions
    #     with open(rebase_file_path, "r") as file:
    #         rebase_instructions = file.readlines()
    #     with open(rebase_file_path, "w") as file:  # modify rebase instructions
    #         first = True
    #         for line in rebase_instructions:
    #             if line.startswith("pick") and not first:
    #                 line = line.replace("pick", "squash", 1)
    #             first = False
    #             file.write(line)
    #     # Apply messages to the last commit
    #     repo.git.commit("--amend","-m",combined_messages)
    #     # Continue to rebase
    #     repo.git.rebase("--continue")

    # def setup_template(self, template_yamlfile,clock_options):
    #     """Setup the template by 
    #        reading YAML file,
    #        checking out the branch,
    #        enabling metadata in config.yaml of the template,
    #        loading param_dict, commt_dict
    #     """
    #     self.expt_yaml = self._read_ryaml(template_yamlfile)
    #     template = self.expt_yaml["template"]
    #     self.template_path = os.path.join(self.dir_manager, template)
    #     repo = git.Repo(self.template_path)
    #     repo.git.checkout(BRANCH_NAME)
    #     if BRANCH_NAME_BASE not in repo.branches:
    #         base_branch = repo.create_head(BRANCH_NAME_BASE)
    #     else:
    #         base_branch = repo.branches[BRANCH_NAME_BASE]
    #     base_branch.checkout()
    #     print(f"Checkout the base branch: {BRANCH_NAME_BASE}")

    #     # Update nuopc clock options, can add more options, TODO
    #     nuopc_path = os.path.join(self.template_path,"nuopc.runconfig")
    #     self.update_nuopc_config(nuopc_path, clock_options)

    #     # Update metadata to true
    #     template_config_data = self._read_ryaml(os.path.join(self.template_path,"config.yaml"))
    #     template_config_data["metadata"]["enable"] = True
    #     self._write_ryaml(os.path.join(self.template_path,'config.yaml'),template_config_data)

    #     # Check file changes
    #     changed_files = self._get_changed_files(repo)
    #     if changed_files:
    #         print("Configure base in preparation for expt runs!")
    #         repo.index.add(changed_files)
    #         commit_message = "Configure base in preparation for expt runs!"
    #         repo.index.commit(commit_message)
    #     else:
    #         print("Nothing changed, hence no further commits to the template repo!")
            
    #     self.startfrom = str(self.expt_yaml['startfrom']).strip().lower().zfill(3)
    #     name_dict = self.expt_yaml["namelists"]["MOM_list"] # TODO make it flexible
    #     param_dict = self._parser_mom6_input(os.path.join(self.template_path, "MOM_input")).param_dict
    #     commt_dict = self._parser_mom6_input(os.path.join(self.template_path, "MOM_input")).commt_dict
    #     self._update_param_dict_change(name_dict, param_dict, commt_dict)

    # def update_nuopc_config(self,nuopc_path, clock_options):
    #     """
    #     Updates lock options.
    #     """
    #     nuopc_config = read_nuopc_config(nuopc_path)
    #     clock_attributes = nuopc_config.get("CLOCK_attributes", {})
    #     for k, v in clock_options.items():
    #         if k in ["stop_option", "stop_n", "restart_option", "restart_n"]:
    #             clock_attributes[k] = v
    #     write_nuopc_config(nuopc_config, nuopc_path)
        

if __name__ == "__main__":
    expt_manager = Expts_manager()
    expt_manager.main()

