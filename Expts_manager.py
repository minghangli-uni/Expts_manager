#!/usr/bin/env python3

import re,os,sys,copy
import git,subprocess
import shutil
import glob
import argparse

try:
    import yaml
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True
except ImportError:
    print("\nFatal error: modules not available.")
    print("On NCI, do the following and try again:")
    print("   module use /g/data/vk83/modules; module load payu\n")
    raise

DIR_MANAGER     = os.getcwd()
DIR_UTILS       = os.path.join(DIR_MANAGER,"tools/om3-utils")
BRANCH_NAME     = "1deg_jra55do_ryf"
BRANCH_NAME_BASE= "_".join([BRANCH_NAME,"baseline"])
BRANCH_PERTURB  = "perturbation"
TEST_REL_PATH   = "test"
EXPT_REL_PATH   = "/".join([TEST_REL_PATH,BRANCH_NAME])

try:
    # currently imported from a fork: https://github.com/minghangli-uni/om3-utils,
    # will update the tool when it is merged to COSIMA/om3-utils
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
        self.indata = None

    def setup_template_through_payu(self,yamlfile):
        self.indata = self._read_ryaml(yamlfile)
        template = self.indata["template"]
        template_branch_name = self.indata["template_branch_name"]
        base_branch_name = self.indata["base_branch_name"]
        base_rel_path = '_'.join([template_branch_name,base_branch_name])
        test_path = self.indata["test_path"]
        
        if os.path.exists(test_path):
            print(f"test directory {test_path} already exists!")    
        else:
            os.makedirs(test_path)
            print(f"test directory {test_path} is created!")
        
        base_path = os.path.join(test_path,base_rel_path)
        
        if os.path.exists(base_path):
            print(f"Base path is already created located at {base_path}")
        else:
            print(f"template ctrl clone {template_branch_name} from {template} to {base_rel_path} as a branch of {base_branch_name}!")
            command = f"payu clone -B {template_branch_name} -b {base_branch_name} {template} {base_path}"
            subprocess.run(command, shell=True, check=False)
            print(f"Baseline is created located at {base_path}")

        # Update nuopc.runconfig
        nuopc_input = self.indata["nuopc_runconfig"]
        if nuopc_input is not None:
            nuopc_file_path = os.path.join(base_path,"nuopc.runconfig")
            nuopc_runconfig = read_nuopc_config(nuopc_file_path)
            self.update_configs(nuopc_runconfig,nuopc_input)
            write_nuopc_config(nuopc_runconfig, nuopc_file_path)
            
        # Update config.yaml
        config_yaml_input = self.indata["config_yaml"]
        if config_yaml_input is not None:
            config_yaml_file = os.path.join(base_path,"config.yaml")
            config_yaml = self._read_ryaml(config_yaml_file)
            self.update_configs(config_yaml,config_yaml_input)
            self._write_ryaml(config_yaml_file,config_yaml)

        # Update coupling timestep through nuopc.runseq
        cpl_dt_input = self.indata["cpl_dt"]
        if cpl_dt_input is not None:
            nuopc_runseq_file = os.path.join(base_path,"nuopc.runseq")
            self.update_cpl_dt_nuopc_seq(nuopc_runseq_file,cpl_dt_input)

        # check file changes
        repo = git.Repo(base_path)
        print(f"Current base branch is: {repo.active_branch.name}")
        changed_files = self._get_changed_files(repo)
        if changed_files:
            print(f"Configure '{base_branch_name}' branch in preparation for expt runs!")
            repo.index.add(changed_files)
            commit_message = f"Configure '{base_branch_name}' branch in preparation for expt runs!"
            repo.index.commit(commit_message)
        else:
            print(f"Nothing changed, hence no further commits to the {base_path} repo!")

    def update_configs(self,base,change):
        """ recursively update nuopc_runconfig and config.yaml entries """
        for k,v in change.items():
            if isinstance(v,dict) and k in base:
                self.update_configs(base[k],v)
            else:
                base[k] = v

    def update_cpl_dt_nuopc_seq(self,seq_path,update_cpl_dt):
        with open(seq_path,'r') as f:
            lines = f.readlines()
        #print(lines)
        
        pattern = re.compile(r'@(\S*)')
        update_lines = []
        for l in lines:
            matches = pattern.findall(l)
            if matches:
                update_line = re.sub(r'@(\S+)', f'@{update_cpl_dt}', l)
                update_lines.append(update_line)
            else:
                update_lines.append(l)
        with open(seq_path,'w') as f:
            f.writelines(update_lines)
            
    
    def setup_template(self, template_yamlfile,clock_options):
        """Setup the template by 
           reading YAML file,
           checking out the branch,
           enabling metadata in config.yaml of the template,
           loading param_dict, commt_dict
        """
        self.indata = self._read_ryaml(template_yamlfile)
        template = self.indata["template"]
        self.template_path = os.path.join(self.dir_manager, template)
        repo = git.Repo(self.template_path)
        repo.git.checkout(BRANCH_NAME)
        if BRANCH_NAME_BASE not in repo.branches:
            base_branch = repo.create_head(BRANCH_NAME_BASE)
        else:
            base_branch = repo.branches[BRANCH_NAME_BASE]
        base_branch.checkout()
        print(f"Checkout the base branch: {BRANCH_NAME_BASE}")

        # Update nuopc clock options, can add more options, TODO
        nuopc_path = os.path.join(self.template_path,"nuopc.runconfig")
        self.update_nuopc_config(nuopc_path, clock_options)

        # Update metadata to true
        template_config_data = self._read_ryaml(os.path.join(self.template_path,"config.yaml"))
        template_config_data["metadata"]["enable"] = True
        self._write_ryaml(os.path.join(self.template_path,'config.yaml'),template_config_data)

        # Check file changes
        changed_files = self._get_changed_files(repo)
        if changed_files:
            print("Configure base in preparation for expt runs!")
            repo.index.add(changed_files)
            commit_message = "Configure base in preparation for expt runs!"
            repo.index.commit(commit_message)
        else:
            print("Nothing changed, hence no further commits to the template repo!")
            
        self.startfrom = str(self.indata['startfrom']).strip().lower().zfill(3)
        name_dict = self.indata["namelists"]["MOM_list"] # TODO make it flexible
        param_dict = self._parser_mom6_input(os.path.join(self.template_path, "MOM_input")).param_dict
        commt_dict = self._parser_mom6_input(os.path.join(self.template_path, "MOM_input")).commt_dict
        self._update_param_dict_change(name_dict, param_dict, commt_dict)

    def update_nuopc_config(self,nuopc_path, clock_options):
        """
        Updates lock options.
        """
        nuopc_config = read_nuopc_config(nuopc_path)
        clock_attributes = nuopc_config.get("CLOCK_attributes", {})
        for k, v in clock_options.items():
            if k in ["stop_option", "stop_n", "restart_option", "restart_n"]:
                clock_attributes[k] = v
        write_nuopc_config(nuopc_config, nuopc_path)

    def _parser_mom6_input(self, path_file):
        mom6parser = MOM6InputParser.MOM6InputParser()
        mom6parser.read_input(path_file)
        mom6parser.parse_lines()
        return mom6parser
        
    def _update_param_dict_change(self, name_dict, param_dict, commt_dict):
        num_expts = len(next(iter(name_dict.values())))  # number of expts
        keys = name_dict.keys()
        param_dict_change_full_list = []  # A list includes full parameter input dicts for all tests
        param_dict_change_list      = []  # A list includes only changed parameter input dicts for all tests
        commt_dict_change           = {key: commt_dict.get(key,'') for key in keys} 

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
        self.commt_dict_change = commt_dict_change
        
    def manage_expts(self):
        for i in range(len(self.param_dict_change_list)):
            # for each experiment
            expt_name = '_'.join([f"{k}_{v}" for k,v in self.param_dict_change_list[i].items()]) 
            rel_path  = '_'.join([EXPT_REL_PATH,expt_name])
            expt_path = os.path.join(self.dir_manager,rel_path)
            print(f"\n {expt_path}")

            if os.path.exists(expt_path):
                print("-- not creating ", rel_path, " - already exists!")
                
            else:
                print("clone template - payu clone!")
                command = f"payu clone -B {BRANCH_NAME_BASE} -b {BRANCH_PERTURB} {self.template_path} {expt_path}" # automatically leave a commit with expt uuid
                test = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                
                # apply changes and write them to `MOM_override`
                MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, "MOM_override"))
                override_param_dict_change, override_commt_dict_change = update_MOM6_params_override(self.param_dict_change_list[i],self.commt_dict_change)
                MOM6_or_parser.param_dict = override_param_dict_change
                MOM6_or_parser.commt_dict = override_commt_dict_change
                MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))
                
                # Update config.yaml 
                config_path = os.path.join(expt_path,"config.yaml")
                config_data = self._read_ryaml(config_path)
                config_data["jobname"] = "_".join([BRANCH_NAME_BASE,expt_name])
                self._write_ryaml(config_path,config_data)
    
                # Update metadata.yaml
                metadata_path = os.path.join(expt_path, "metadata.yaml")
                metadata = self._read_ryaml(metadata_path)
                self._update_metadata_description(metadata)
                self._remove_metadata_comments("description", metadata)
                keywords = self._extract_metadata_keywords(self.param_dict_change_list[i])
                metadata["keywords"] = f"{BRANCH_NAME_BASE}, {BRANCH_PERTURB}, {keywords}"
                self._remove_metadata_comments("keywords", metadata)
                self._write_ryaml(metadata_path, metadata)

                # replace metadata in archive/
                shutil.copy(metadata_path,os.path.join(expt_path,"archive"))
    
                repo = git.Repo(expt_path)
                
                changed_files = self._get_changed_files(repo)
                untracked_files = self._get_untracked_files(repo)
                files_to_stages = set(changed_files+untracked_files)
                
                if files_to_stages:  # commit changes for the expt runs
                    print(f"files need to be staged: {files_to_stages}")
                    repo.index.add(files_to_stages)
                    commit_message = f"Payu clone from the base: {self.template_path}; committed files/directories are: {', '.join(f'{i}' for i in files_to_stages)}"
                    repo.index.commit(commit_message)
                else:
                    print("No files are required to be committed!")

            if self.startfrom == "rest":
                print("start perturbation from rest!")
            else:  # WORKING ON
                restart_path = os.path.join(expt_path,"archive","restart",self.startfrom)

            if self.indata["nruns"] > 0:
                doneruns = len(glob.glob(os.path.join(expt_path,"archive","output[0-9][0-9][0-9]*")))
                newruns = self.indata["nruns"] - doneruns
                if newruns > 0:
                    command = f"cd {expt_path} && payu run -n {newruns}"
                    subprocess.run(command, check=False, shell=True)
                else:
                    print(f"{expt_path} has already completed {doneruns} runs! Hence stop without running!")

                
    def _get_untracked_files(self,repo):
        return repo.untracked_files
    
    def _get_changed_files(self,repo):
        return [file.a_path for file in repo.index.diff(None)]   
                
    def _read_ryaml(self, yaml_path):
        """ Read yaml file."""
        with open(yaml_path, "r") as f:
            return ryaml.load(f)
            
    def _write_ryaml(self,yaml_path,data):
        """ Write yaml file."""
        with open(yaml_path, "w") as f:
            ryaml.dump(data,f)
            
    def _update_metadata_description(self, metadata):
        """Update metadata description with experiment details."""
        desc = metadata["description"]
        if desc is None:
            desc = ''
        desc += (f'\nNOTE: this is a perturbation experiment, but the description above is for the control run.'
                f'\nThis perturbation experiment is based on the control run {self.template_path}')
        if self.startfrom == "rest":
            desc += "\nbut with condition of rest."
        metadata["description"] = desc

    def _remove_metadata_comments(self, key, metadata):
        """Remove comments after the key in metadata."""
        if key in metadata:
            metadata.ca.items[key] = [None, None, None, None]

    def _extract_metadata_keywords(self, param_change_dict):
        """Extract keywords from parameter change dictionary."""
        keywords = ', '.join(param_change_dict.keys())
        return keywords

    def _get_previous_commit_messages(self,repo, num_commits):  # not used now
        messages = []
        commit = repo.head.commit  # get the head commit
        for _ in range(num_commits):
            messages.append(commit.message.strip())
            if commit.parents:
                commit = commit.parents[0]
            else:
                break
        return messages
        
    def _rebase_and_squash(self, repo, num_commits):    # not used now
        previous_messages = self._get_previous_commit_messages(repo,num_commits) 
        combined_messages =  "\n\n".join(previous_messages)
        repo.git.rebase("-i",f"HEAD~{num_commits}")
        rebase_file_path = os.path.join(repo.git_dir,"/rebase-merge/git-rebase-todo")  # rebase instructions
        with open(rebase_file_path, "r") as file:
            rebase_instructions = file.readlines()
        with open(rebase_file_path, "w") as file:  # modify rebase instructions
            first = True
            for line in rebase_instructions:
                if line.startswith("pick") and not first:
                    line = line.replace("pick", "squash", 1)
                first = False
                file.write(line)
        # Apply messages to the last commit
        repo.git.commit("--amend","-m",combined_messages)
        # Continue to rebase
        repo.git.rebase("--continue")

    def main(self):
        parser = argparse.ArgumentParser(description="Manage ACCESS-OM3 experiments.\
                 Latest version and help: https://github.com/minghangli-uni/Expts_manager")
        parser.add_argument("INPUT_YAML", type=str, nargs="?",default="expt_mom.yaml",
                            help="YAML file specifying parameter values for expt runs. Default is expt_mom.yaml")
        args = parser.parse_args()
        INPUT_YAML = vars(args)["INPUT_YAML"]
        
        clock_options = {"stop_option":"ndays",
                "restart_option":"ndays",
                "stop_n":1,
                "restart_n":1,
                }
        
        yamlfile     = os.path.join(DIR_MANAGER,INPUT_YAML)
        self.setup_template_through_payu(yamlfile)
        #self.setup_template(yamlfile,clock_options)
        #self.manage_expts()

if __name__ == "__main__":
    expt_manager = Expts_manager()
    expt_manager.main()

