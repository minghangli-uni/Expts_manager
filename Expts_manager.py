#!/usr/bin/env python3
"""
A python script to manage ACCESS-OM3 experiment runs

Latest version: 
Author: Minghang Li
Apache 2.0 License http://www.apache.org/licenses/LICENSE-2.0.txt
"""
# ===========================================================================
import os
import sys
import re
import copy
import subprocess
import shutil
import glob
import argparse
import warnings
try:
    import numpy as np
    import git
    import f90nml 
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True
except ImportError:
    print("\nFatal error: modules not available.")
    print("On NCI, do the following and try again:")
    print("   module use /g/data/vk83/modules && module load payu/1.1.3\n")
    raise

# ===========================================================================

def update_MOM6_params_override(param_dict_change, commt_dict_change):
    """Prepends `#override` to parameters for MOM6."""
    override_param_dict_change = {f"#override {k}": v for k,v in param_dict_change.items()}
    override_commt_dict_change = {f"#override {k}": v for k,v in commt_dict_change.items()}
    return override_param_dict_change, override_commt_dict_change

class Expts_manager(object):
    DIR_MANAGER     = os.getcwd()
    BRANCH_PERTURB  = "perturb"

    def __init__(self, MOM_prefix: str = "MOM_list", combo_suffix: str = "_combo", nml_suffix: str = "_nml"):
        self.branch_perturb = self.BRANCH_PERTURB
        self.dir_manager = self.DIR_MANAGER
        self.MOM_prefix = MOM_prefix
        self.combo_suffix = combo_suffix
        self.nml_suffix = nml_suffix
        self.template_path = None
        self.startfrom = None
        self.indata = None
        self.base_branch_name = None
        self.base_path = None
        self.test_path = None
        self.expt_names = None
        self.utils_path = None
        self.utils_dir_name = None
        self.utils_branch_name = None
        self.num_expts = None
        self.param_dict_change_list = []
        self.commt_dict_change = {}
        self.append_group_list = []

        self.previous_key = None
        
    def load_variables(self,yamlfile):
        self.yamlfile = yamlfile  # path for Expts_manager.yaml 
        self.indata = self._read_ryaml(yamlfile)  # load Expts_manager.yaml 
        self.utils_url = self.indata["utils_url"]  # git url for om3-utils tool
        self.utils_dir_name = self.indata["utils_dir_name"]  # user-defined directory for the om3-utils tool
        self.utils_branch_name = self.indata["utils_branch_name"]  # branch name for usage
        self.template_url = self.indata["template_url"]  # template git url
        self.template_commit = self.indata["template_commit"]  # specific git commit 
        self.base_dir_name = self.indata["base_dir_name"]  # user-defined directory for a baseline control experiment
        self.base_branch_name = self.indata["base_branch_name"]  # user-defined branch name for the control experiment
        self.test_path = self.indata["test_path"]  # user-defined path for the test runs, including the control experiment and multiple parameter-tunning experiments
        self.ctrl_nruns = self.indata["ctrl_nruns"]
        self.startfrom = self.indata["startfrom"]  #
        self.startfrom_str = str(self.startfrom).strip().lower().zfill(3)  # define `startfrom_str` following `Expts_manager.yaml`
        self.nruns = self.indata["nruns"]
        self.ice_in_ctrl = None
        self.tag_model = None

    def load_tools(self):
        """Load external tools required for the experiments."""
        # currently imported from a fork: https://github.com/minghangli-uni/om3-utils
        # will update the tool when it is merged to COSIMA/om3-utils
        self.utils_path = os.path.join(self.dir_manager,self.utils_dir_name)
        if not os.path.isdir(self.utils_path):
            command = f"git clone --branch {self.utils_branch_name} {self.utils_url} {self.utils_path} --single-branch"
            subprocess.run(command, shell=True, check=True)
            print(f"Cloning {self.utils_dir_name} for usage!")
        sys.path.append(self.utils_path)
        from om3utils import MOM6InputParser
        from om3utils.nuopc_config import read_nuopc_config, write_nuopc_config
        self.MOM6InputParser = MOM6InputParser
        self.read_nuopc_config = read_nuopc_config
        self.write_nuopc_config = write_nuopc_config
        
    def setup_ctrl_expt(self):
        # Create test path
        if os.path.exists(self.test_path):
            print(f"test directory {self.test_path} already exists!")
        else:
            os.makedirs(self.test_path)
            print(f"test directory {self.test_path} is created!")

        # create the path for the control experiment
        self.base_path = os.path.join(self.test_path,self.base_dir_name)
        
        # Clone the control repo from a designated repo defined in `Expts_manager.yaml` and create a new branch
        if os.path.exists(self.base_path):
            print(f"Base path is already created located at {self.base_path}")
        else:
            print(f"Cloning template from {self.template_url} to {self.base_path}")
            command = f"payu clone {self.template_url} {self.base_path}"
            subprocess.run(command, shell=True, check=False)
            templaterepo = git.Repo(self.base_path)
            print(f"Checkout commit {self.template_commit} - corresponding to repo's branch: {templaterepo.active_branch.name};")
            print(f"and create a new branch {self.base_branch_name} for the control experiment!")
            templaterepo.git.checkout('-b', self.base_branch_name, self.template_commit)  # checkout the new branch from the specific template commit

        # Update configuration files, including `nuopc.runconfig`, `config.yaml`, only coupling timestep from `nuopc.runseq`
        self._update_config_files()

        # load ice_in of the control experiment
        self.ice_in_ctrl = f90nml.read(os.path.join(self.base_path,"ice_in"))

        # Payu setup && sweep to ensure the changes correctly && remove the `work` directory for the control run
        command = f"cd {self.base_path} && payu setup && payu sweep"
        subprocess.run(command, shell=True, check=False)
        
        # check file changes and commits if so, otherwise, no commits for the ctrl run.
        doneruns = len(glob.glob(os.path.join(self.base_path,"archive","output[0-9][0-9][0-9]*")))
        if doneruns == 0:
            repo = git.Repo(self.base_path)
            print(f"Current base branch is: {repo.active_branch.name}")
            untracked_files = self._get_untracked_files(repo)
            changed_files = self._get_changed_files(repo)
            staged_files = set(untracked_files+changed_files)
            commit_message = f"Control experiment setup: Configure `{self.base_branch_name}` branch by `{self.yamlfile}`\nin preparation for expt runs!"
            if staged_files:
                repo.index.add(staged_files)
                repo.index.commit(commit_message)
            else:
                print(f"Nothing changed, hence no further commits to the {self.base_path} repo!")

        # run the control experiment
        if self.ctrl_nruns > 0:
            print(f"\nRun control experiment -n {self.ctrl_nruns}\n")
            command = f"cd {self.base_path} && payu run -f -n {self.ctrl_nruns}"
            subprocess.run(command, check=False, shell=True)
        else:
            print(f"\nNo new control experiments running!\n")

    def setup_perturb_expt(self):
        # Check namelists in `Expts_manager.yaml`
        namelists = self.indata["namelists"]
        if namelists is not None:
            print("==== Perturbation experiments ====")
            for k,nmls in namelists.items(): 
                if k.startswith('CICE_input'):
                    self.tag_model = 'cice'
                    cice_list_count = 0
                    for k_sub in nmls:
                        if k_sub.endswith(self.nml_suffix) or k_sub.endswith(self.combo_suffix):
                            name_dict = nmls[k_sub]
                            if k_sub.endswith(self.combo_suffix):
                                if isinstance(next(iter(name_dict.values())), list):
                                    self.num_expts = len(next(iter(name_dict.values())))
                                else:
                                    self.num_expts = 1
                            else:
                                self.num_expts = 0
                                for v_s in name_dict.values():
                                    if isinstance(v_s,list):
                                        self.num_expts += len(v_s)
                                    else:
                                        self.num_expts = 1
                            cice_list_count += 1

                            if self.previous_key is not None and self.previous_key.startswith("CICE_expt_dir"):
                                self.expt_names = nmls[self.previous_key]
                                if self.expt_names is not None:
                                    if len(self.expt_names) != self.num_expts:
                                        raise ValueError(f"The number of user-defined experiment directories {self.expt_names} "
                                                         f"is different from that of tunning parameters {name_dict}!"
                                                         f"\nPlease double check the number or leave it blank!") 
                            commt_dict = None  # only valids for MOM_input
                            if k_sub.endswith(self.combo_suffix):
                                self._generate_combined_dicts(name_dict,commt_dict,k_sub)
                            else:
                                self._generate_individual_dicts(name_dict,commt_dict,k_sub)
                            self.manage_expts()
                        self.previous_key = k_sub      
                if k.startswith("MOM_input"):
                    self.tag_model = 'mom6'
                    mom_list_count = 0
                    for k_sub in nmls:
                        if k_sub.startswith(self.MOM_prefix):
                            name_dict = nmls[k_sub]
                            if k_sub.endswith(self.combo_suffix):
                                if isinstance(next(iter(name_dict.values())), list):
                                    self.num_expts = len(next(iter(name_dict.values())))
                                else:
                                    self.num_expts = 1
                            else:
                                self.num_expts = 0
                                for v_s in name_dict.values():
                                    if isinstance(v_s,list):
                                        self.num_expts += len(v_s)
                                    elif isinstance(v_s,bool):
                                        self.num_expts = 1
                            mom_list_count += 1
                            if self.previous_key is not None and self.previous_key.startswith("MOM_expt_dir"):
                                self.expt_names = nmls[self.previous_key]
                                if self.expt_names is not None:
                                    if len(self.expt_names) != self.num_expts:
                                        raise ValueError(f"The number of user-defined experiment directories {self.expt_names} "
                                                         f"is different from that of tunning parameters {name_dict}!"
                                                         f"\nPlease double check the number or leave it blank!") 
                            MOM_inputParser = self._parser_mom6_input(os.path.join(self.base_path, "MOM_input"))
                            param_dict = MOM_inputParser.param_dict
                            commt_dict = MOM_inputParser.commt_dict
                            if k_sub.endswith(self.combo_suffix):
                                self._generate_combined_dicts(name_dict,commt_dict,k_sub)
                            else:
                                self._generate_individual_dicts(name_dict,commt_dict,k_sub)
                            self.manage_expts()
                        self.previous_key = k_sub
        else:
#            raise ValueError("namelists can't be None for paramater-tunning tests!")
             warnings.warn("NO namelists provided, hence there are no parameter-tunning tests!")             
    def _generate_individual_dicts(self,name_dict,commt_dict,k_sub):
        """Each dictionary has a single key-value pair."""
        param_dict_change_list = []
        append_group_list = []
        for k, vs in name_dict.items():
            if isinstance(vs, list):
                for v in vs:
                    param_dict_change = {k: v}
                    param_dict_change_list.append(param_dict_change)
                    append_group = k_sub
                    append_group_list.append(append_group)
            else:
                param_dict_change = {k: vs}
                param_dict_change_list.append(param_dict_change)
                append_group_list = [k_sub] 
        self.param_dict_change_list = param_dict_change_list
        if self.tag_model == 'mom6':
            self.commt_dict_change = {k: commt_dict.get(k,"") for k in name_dict}
        elif self.tag_model == 'cice':
            self.append_group_list = append_group_list
        
    def _generate_combined_dicts(self,name_dict,commt_dict,k_sub):
        """Generate a list of dictionaries where each dictionary contains all keys with values from the same index."""
        param_dict_change_list = []
        append_group_list = []
        for i in range(self.num_expts):
            param_dict_change = {k: name_dict[k][i] for k in name_dict}
            append_group = k_sub
            append_group_list.append(append_group)
            param_dict_change_list.append(param_dict_change)
        self.param_dict_change_list = param_dict_change_list
        if self.tag_model == 'mom6':
            self.commt_dict_change = {k: commt_dict.get(k,"") for k in name_dict}
        elif self.tag_model == 'cice':
            self.append_group_list = append_group_list
            
    def manage_expts(self):
        """ setup expts, and run expts"""
        for i in range(len(self.param_dict_change_list)):
            # for each experiment
            param_dict = self.param_dict_change_list[i]
            if self.tag_model == 'cice':
                cice_group = self.append_group_list[i]
            if self.expt_names is None:  
                expt_name = "_".join([f"{k}_{v}" for k,v in self.param_dict_change_list[i].items()])  # if `expt_names` does not exist, expt_name is set as the tunning parameters appending with associated values
            else:
                expt_name = self.expt_names[i]  # user-defined folder names for parameter-tunning experiments
            rel_path = os.path.join(self.test_path,expt_name)
            expt_path = os.path.join(self.dir_manager,rel_path)

            if os.path.exists(expt_path):
                print("-- not creating ", expt_path, " - already exists!")
            else:
                # create perturbation experiment - check if needs skipping!
                if self.tag_model == 'cice':
                    if cice_group.endswith(self.combo_suffix):  # rename the namlist if suffix with `_combo`
                        cice_group = cice_group[:-len(self.combo_suffix)]
                
                    cice_name = param_dict.keys()
                    if len(cice_name) == 1:  # one param:value pair
                        cice_value = param_dict[list(cice_name)[0]]
                    else:  # combination of param:value pairs
                        cice_value = [param_dict[j] for j in cice_name]
                    
                    if 'turning_angle' in param_dict:
                        cosw = np.cos(param_dict['turning_angle'] * np.pi / 180.)
                        sinw = np.sin(param_dict['turning_angle'] * np.pi / 180.)
                        
                    if all(cn in self.ice_in_ctrl.get(cice_group,{}) for cn in cice_name):  # cice_name (i.e. tunning parameter) may not be found in the control experiment
                        if 'turning_angle' in param_dict:
                            skip = (self.ice_in_ctrl[cice_group]['cosw'] == cosw and 
                                    self.ice_in_ctrl[cice_group]['sinw'] == sinw and
                                    all(self.ice_in_ctrl[cice_group].get(cn) == param_dict[cn] for cn in cice_name if cn not in ['cosw', 'sinw']))
                        else:
                            skip = all(self.ice_in_ctrl[cice_group].get(cn) == param_dict[cn] for cn in cice_name)
                    else:
                        print(f"Not all {cice_name} are found in {cice_group}, hence not skipping!")
                        skip = False
    
                    if skip:
                        print('-- not creating', expt_path, '- parameters are identical to the control experiment located at', self.base_path)
                        continue
                    
                if self.tag_model == 'mom6': # might need MOM_parameter.all, because many parameters are in-default and not shown up in `MOM_input` 
                    #TODO
                    pass
                    
                print(f"Directory {expt_path} not exists, hence cloning template!")
                command = f"payu clone -B {self.base_branch_name} -b {self.branch_perturb} {self.base_path} {expt_path}" # automatically leave a commit with expt uuid
                subprocess.run(command, shell=True, check=True)

            # Update `MOM_override` or/and `ice_in`
            if self.tag_model == 'mom6':
                # apply changes and write them to `MOM_override`
                MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, "MOM_override"))  # parse MOM_override
                MOM6_or_parser.param_dict, MOM6_or_parser.commt_dict = update_MOM6_params_override(param_dict,self.commt_dict_change)  # update the tunning parameters, values and associated comments
                MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))  # write to file
                
            elif self.tag_model == 'cice':
                # do changes
                cice_path = os.path.join(expt_path,"ice_in")
                if cice_group.endswith(self.combo_suffix):  # rename the namlist if suffix with `_combo`
                    cice_group = cice_group[:-len(self.combo_suffix)]
                patch_dict = {cice_group: {}}
                for cice_name, cice_value in param_dict.items():
                    if cice_name == 'turning_angle':
                        cosw = np.cos(cice_value * np.pi / 180.)
                        sinw = np.sin(cice_value * np.pi / 180.)
                        patch_dict[cice_group]['cosw'] = cosw
                        patch_dict[cice_group]['sinw'] = sinw
                    else:  # for generic parameters
                        patch_dict[cice_group][cice_name] = cice_value
                f90nml.patch(cice_path, patch_dict, cice_path+'_tmp')
                os.rename(cice_path+'_tmp',cice_path)

            if self.startfrom_str != 'rest':  # symlink restart directories
                link_restart = os.path.join('archive','restart'+self.startfrom_str)  # 
                restartpath = os.path.realpath(os.path.join(self.base_path,link_restart))  # restart directory from control experiment
                dest = os.path.join(expt_path, link_restart)  # restart directory symlink for each perturbation experiment
                if not os.path.islink(dest):  # if symlink exists, skip creating the symlink
                    os.symlink(restartpath, dest)  # done the symlink if does not exist
                    
            # Update config.yaml 
            config_path = os.path.join(expt_path,"config.yaml")
            config_data = self._read_ryaml(config_path)
            config_data["jobname"] = expt_name
            self._write_ryaml(config_path,config_data)

            # Update metadata.yaml
            metadata_path = os.path.join(expt_path, "metadata.yaml")  # metadata path for each perturbation
            metadata = self._read_ryaml(metadata_path)  # load metadata of each perturbation
            self._update_metadata_description(metadata,restartpath)  # update `description`
            self._remove_metadata_comments("description", metadata)  # remove None comments from `description`
            keywords = self._extract_metadata_keywords(param_dict)  # extract parameters from the change list
            metadata["keywords"] = f"{self.base_dir_name}, {self.branch_perturb}, {keywords}"  # update `keywords`
            self._remove_metadata_comments("keywords", metadata)  # remove None comments from `keywords`
            self._write_ryaml(metadata_path, metadata)  # write to file

            # clean `work` directory for failed jobs
            self._clean_workspace(expt_path)

            # commit the above changes for the expt runs   
            doneruns = len(glob.glob(os.path.join(expt_path,"archive","output[0-9][0-9][0-9]*")))    
            if doneruns == 0:
                exptrepo = git.Repo(expt_path)
                untracked_files = self._get_untracked_files(exptrepo)
                changed_files = self._get_changed_files(exptrepo)
                staged_files = set(untracked_files+changed_files)
                self._restore_swp_files(exptrepo,staged_files)  # restore *.swp files in case users open any files during case is are running
                if staged_files:  # commit changes for each expt run
                    exptrepo.index.add(staged_files)
                    commit_message = f"Experiment setup: Clone from the control experiment: {self.base_path};\nCommitted files/directories are: {', '.join(f'{j}' for j in staged_files)}\n"
                    exptrepo.index.commit(commit_message)
                    print(f"files have been committed: {staged_files}\n")

            # start runs, count existing runs and do additional runs if needed
            if self.nruns > 0:
                newruns = self.nruns - doneruns
                if newruns > 0:
                    command = f"cd {expt_path} && payu run -n {newruns}"
                    subprocess.run(command, check=False, shell=True)
                    print('\n')
                else:
                    print(f"-- `{expt_name}` has already completed {doneruns} runs! Hence stop running!\n")
                    
        self.expt_names = None  # reset to None after the loop to update user-defined perturbation experiment names!
        
    def _clean_workspace(self,dir_path):
        work_dir = os.path.join(dir_path,'work')
        if os.path.islink(work_dir) and os.path.isdir(work_dir):  # in case any failed job
            # Payu sweep && setup to ensure the changes correctly && remove the `work` directory
            command = f"payu sweep && payu setup"
            subprocess.run(command, shell=True, check=False)
            print(f"Clean up a failed job {work_dir} and prepare it for resubmission.")
            
    def _parser_mom6_input(self, path_file):
        """ parse MOM6 input file """
        mom6parser = self.MOM6InputParser.MOM6InputParser()
        mom6parser.read_input(path_file)
        mom6parser.parse_lines()
        return mom6parser

    def _update_config_files(self):    
        """Update configuration files based on YAML settings."""
        # Update nuopc.runconfig for the ctrl run
        nuopc_input = self.indata["nuopc_runconfig"]
        if nuopc_input is not None:
            nuopc_file_path = os.path.join(self.base_path,"nuopc.runconfig")
            nuopc_runconfig = self.read_nuopc_config(nuopc_file_path)
            self._update_config_entries(nuopc_runconfig,nuopc_input)
            self.write_nuopc_config(nuopc_runconfig, nuopc_file_path)

        # Update config.yaml for the ctrl run
        config_yaml_input = self.indata["config_yaml"]
        if config_yaml_input is not None:
            config_yaml_file = os.path.join(self.base_path,"config.yaml")
            config_yaml = self._read_ryaml(config_yaml_file)
            self._update_config_entries(config_yaml,config_yaml_input)
            self._write_ryaml(config_yaml_file,config_yaml)

        # Update coupling timestep through nuopc.runseq for the ctrl run
        cpl_dt_input = self.indata["cpl_dt"]
        if cpl_dt_input is not None:
            nuopc_runseq_file = os.path.join(self.base_path,"nuopc.runseq")
            self._update_cpl_dt_nuopc_seq(nuopc_runseq_file,cpl_dt_input)

    def _update_config_entries(self,base,change):
        """ recursively update nuopc_runconfig and config.yaml entries """
        for k,v in change.items():
            if isinstance(v,dict) and k in base:
                self._update_config_entries(base[k],v)
            else:
                base[k] = v

    def _update_cpl_dt_nuopc_seq(self,seq_path,update_cpl_dt):
        """ update only coupling timestep through nuopc.runseq"""
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

    def _get_untracked_files(self,repo):
        """ get untracked git files """
        return repo.untracked_files

    def _get_changed_files(self,repo):
        """ get changed git files """
        return [file.a_path for file in repo.index.diff(None)]   

    def _get_deleted_files(self,repo):
        return [file.a_path for file in repo.index.diff(None) if file.deleted_file]
        
    def _restore_swp_files(self,repo,staged_files):
        swp_files = [file for file in staged_files if file.endswith('.swp')]
        for file in swp_files:
            repo.git.restore(file)

    def _read_ryaml(self, yaml_path):
        """ Read yaml file and preserve comments"""
        with open(yaml_path, "r") as f:
            return ryaml.load(f)

    def _write_ryaml(self,yaml_path,data):
        """ Write yaml file and preserve comments"""
        with open(yaml_path, "w") as f:
            ryaml.dump(data,f)

    def _update_metadata_description(self, metadata,restartpath):
        """Update metadata description with experiment details."""
        tmp_string1 = (f"\nNOTE: this is a perturbation experiment, but the description above is for the control run."
                    f"\nThis perturbation experiment is based on the control run\n {self.base_path} from {self.base_branch_name}")
        tmp_string2 = f"\nbut with initial condition {restartpath}."
        desc = metadata["description"]
        if desc is None:
            desc = ""
        if tmp_string1.strip() not in desc.strip():
            desc += tmp_string1
        if tmp_string2.strip() not in desc.strip():
            desc += tmp_string2
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

        yamlfile     = os.path.join(self.dir_manager,INPUT_YAML)
        self.load_variables(yamlfile)
        self.load_tools()
        self.setup_ctrl_expt()
        self.setup_perturb_expt()
        
if __name__ == "__main__":
    expt_manager = Expts_manager()
    expt_manager.main()

