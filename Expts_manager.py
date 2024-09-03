#!/usr/bin/env python3
"""
ACCESS-OM3 Experiment Management Tool
This python script manages experiment runs for ACCESS-OM3, providing functionalities
to set up control and perturbation experiments, modify configuration files,
and manage related utilities.

Latest version: https://github.com/minghangli-uni/Expts_manager
Author: Minghang Li
License: Apache 2.0 License http://www.apache.org/licenses/LICENSE-2.0.txt
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
    print("   module use /g/data/vk83/modules && module load payu/1.1.5\n")
    raise

# ===========================================================================

class LiteralString(str):
    pass
def represent_literal_str(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

ryaml.representer.add_representer(LiteralString, represent_literal_str)

def update_MOM6_params_override(param_dict_change, commt_dict_change):
    """
    Prepends `#override` to parameters for MOM6.
    Args:
        param_dict_change (dict): dictionary of parameters to override
        commt_dict_change (dict): dictionary of comments for parameters
    Returns:
        tuple: Two dictionaries with `#override` prepended to each key.
    """
    
    override_param_dict_change = {f"#override {k}": v for k,v in param_dict_change.items()}
    override_commt_dict_change = {f"#override {k}": v for k,v in commt_dict_change.items()}
    return override_param_dict_change, override_commt_dict_change

class Expts_manager(object):
    """
    A class to manage ACCESS-OM3 experiment runs, including control and perturbation experiments
    Attributes:
        combo_suffix (str): Suffix for combo experiments, i.e., multiple-parameter test
        branch_perturb (str): user-defined branch name for the perturbation
    """
    DIR_MANAGER     = os.getcwd()

    def __init__(self,
                 MOM_prefix: str='MOM_list',
                 runseq_prefix: str='runseq_list',
                 combo_suffix: str="_combo",
                 branch_perturb: str="perturb"):

        self.dir_manager = self.DIR_MANAGER
        self.MOM_prefix = MOM_prefix
        self.runseq_prefix = runseq_prefix
        self.branch_perturb = branch_perturb
        self.combo_suffix = combo_suffix
        
    
    def load_variables(self,yamlfile):
        """ 
        Load variables from the input yaml file 
        Args:
            yamlfile (str): Path to the yaml configuration file, i.e., Expts_manager.yaml 
        Attributes:
            yamlfile (str): Path to the yaml configuration file
            indata (dict): Data loaded from the yaml file
            utils_url (str): Git url for the om3-utils tool
            utils_branch_name (str): Branch name for the om3-utils tool
            utils_dir_name (str): User-defined directory for the om3-utils tool
            template_url (dict): Git url for the ACCESS-OM3 configuration
            template_commit (str): Specific git commit for the ACCESS-OM3 configuration
            base_dir_name (str): User-defined directory name for the baseline control experiment
            base_branch_name (str): User-defined branch name for the control experiment
            test_path (str): User-defined path for test runs, including control and perturbation experiments
            startfrom (int/str): Restart number of the control experiment used as an initial condition for perturbation tests; use 'rest' to start from the initial state
            startfrom_str (str): String representation of `startfrom`, padded to three digits
            ctrl_nruns (int): Number of control runs. It is associated with total number of output directories that have been generated
            pert_nruns (int): Number of perturbation experiment runs; associated with total number of output directories that have been generated
        """
        self.yamlfile = yamlfile
        self.indata = self._read_ryaml(yamlfile)
        self.utils_url = self.indata["utils_url"]
        self.utils_dir_name = self.indata["utils_dir_name"]
        self.utils_branch_name = self.indata["utils_branch_name"]
        self.template_url = self.indata["template_url"]
        self.template_commit = self.indata["template_commit"]
        self.base_dir_name = self.indata["base_dir_name"]
        self.base_branch_name = self.indata["base_branch_name"]
        self.test_path = self.indata["test_path"]
        self.startfrom = self.indata["startfrom"]
        self.startfrom_str = str(self.startfrom).strip().lower().zfill(3)
        self.ctrl_nruns = self.indata.get("ctrl_nruns",0)
        self.nruns = self.indata.get("nruns",0)
        self.run_namelists = self.indata.get("run_namelists",False)
        self.check_skipping = self.indata.get("check_skipping",False)
        self.force_restart = self.indata.get("force_restart",False)
        self.diag_url = self.indata.get("diag_url",None)
        self.diag_dir_name = self.indata.get("diag_dir_name",None)
        self.diag_branch_name = self.indata.get("diag_branch_name",None)
        self.diag_ctrl = self.indata.get("diag_ctrl",False)
        self.diag_pert = self.indata.get("diag_pert",False)
        self._initialise_variables()

    def _initialise_variables(self):
        """ 
        Initialise variables from experiment setups
        nml_ctrl (f90nml): f90 namlist for the interested parameters. It is used as a base to modify for perturbation experiments
        tag_model (str): Switch for tuning parameters between f90 namelist and MOM_input
        param_dict_change_list list[dict]: Specific for MOM_input, the list containing tunning parameter dictionaries
        commt_dict_change (dict): Specific for MOM_input, dictionary of comments for parameters
        append_group_list (list): Specific for f90nml, the list containing tunning parameter
        expt_names list[str]: Optional user-defined directory names for perturbation experiments
        """
        self.nml_ctrl = None
        self.tag_model = None
        self.param_dict_change_list = []
        self.commt_dict_change = {}
        self.append_group_list = []
        self.previous_key = None
        self.expt_names = None
        self.diag_path = None

    def load_tools(self):
        """
        Load external tools required for the experiments.
        """
        # currently import from a fork: https://github.com/minghangli-uni/om3-utils
        # will update the tool when it is merged to COSIMA/om3-utils
        def clone_repo(branch_name, url, path, tool_name):
            if not os.path.isdir(path):
                command = f"git clone --branch {branch_name} {url} {path} --single-branch"
                subprocess.run(command, shell=True, check=True)
                print(f"Cloning {tool_name} for use!")
            else:
                print(f"{path} already exists!")
        
        # om3-utils is a must
        utils_path = os.path.join(self.dir_manager, self.utils_dir_name)
        clone_repo(self.utils_branch_name, self.utils_url, utils_path, self.utils_dir_name)

        # make_diag_table is [optional]
        self.diag_path = os.path.join(self.dir_manager, self.diag_dir_name) if self.diag_dir_name else None
        clone_repo(self.diag_branch_name, self.diag_url, self.diag_path, self.diag_dir_name)

        sys.path.extend([utils_path, self.diag_path])

        # load modules from om3-utils
        from om3utils import MOM6InputParser
        from om3utils.nuopc_config import read_nuopc_config, write_nuopc_config
        self.MOM6InputParser = MOM6InputParser
        self.read_nuopc_config = read_nuopc_config
        self.write_nuopc_config = write_nuopc_config

    def create_test_path(self):
        """
        Create the local test directory for blocks of parameter testing.
        """
        if os.path.exists(self.test_path):
            print(f"test directory {self.test_path} already exists!")
        else:
            os.makedirs(self.test_path)
            print(f"test directory {self.test_path} is created!")
         
    def manage_ctrl_expt(self):
        """
        Setup and run the control experiment
        """
        self.base_path = os.path.join(self.test_path,self.base_dir_name)
        base_path = self.base_path
        ctrl_nruns = self.ctrl_nruns
        
        if os.path.exists(base_path):
            print(f"Base path is already created and located at {base_path}")
            if self._count_file_nums() == 4:
                print("previous commit fails, please try with an updated commit hash for the control experiment!")
                self._extract_config_via_commit()
                if self.diag_ctrl:
                    self._copy_diag_table(base_path)
                self._setup_ctrl_expt()
                self._check_and_commit_changes()
        else:
            self._clone_template_repo()  # clone the template repo and setup the control branch
            self._extract_config_via_commit()
            if self.diag_ctrl:
                self._copy_diag_table(base_path)
            self._setup_ctrl_expt()
            self._check_and_commit_changes()

        # run the control experiment
        if ctrl_nruns > 0:
            doneruns = len(glob.glob(os.path.join(base_path,"archive","output[0-9][0-9][0-9]*")))  # check the number of existing output directories
            newruns = ctrl_nruns - doneruns
            if newruns > 0:
                print(f"\nRun control experiment -n {newruns}\n")
                command = f"cd {base_path} && payu run -f -n {newruns}"
                subprocess.run(command, check=False, shell=True)
            else:
                print(f"ctrl_nruns ({ctrl_nruns}) equals to the number of existing output directories ({doneruns}), hence no new control experiments will start!\n")
        else:
            print(f"ctrl_nruns is {ctrl_nruns}, hence no new control experiments will start!\n")

    def _clone_template_repo(self):
        """
        Clone the template repo.
        """
        print(f"Cloning template from {self.template_url} to {self.base_path}")
        command = f"payu clone {self.template_url} {self.base_path}"
        subprocess.run(command, shell=True, check=False)

    def _extract_config_via_commit(self):
        """
        Extract specific configuration via commit hash.
        """
        templaterepo = git.Repo(self.base_path)
        print(f"Checking out commit {self.template_commit} and creating new branch {self.base_branch_name}!")
        templaterepo.git.checkout('-b', self.base_branch_name, self.template_commit)  # checkout the new branch from the specific template commit

    def _copy_diag_table(self,path):
        """
        Copy the diagnostic table (`diag_table`) to the specified path if a path is defined.
        """
        if self.diag_path:
            command = f"scp {os.path.join(self.diag_path,'diag_table')} {path}"
            subprocess.run(command, shell=True, check=False)
            print(f"Copy diag_table to {path}")
        else:
            print(f"{self.diag_path} is not defined, hence skip copy diag_table to the control experiment")

    def _count_file_nums(self):
        """
        Count the number of file numbers.
        """
        return len(os.listdir(self.base_path))

    def _setup_ctrl_expt(self):
        # Update configuration files, namelist and MOM_input for the control experiment if needed.
        self._update_contrl_params()

        # Payu setup && sweep to ensure the changes correctly && remove the `work` directory for the control run
        command = f"cd {self.base_path} && payu setup && payu sweep"
        subprocess.run(command, shell=True, check=False)

    def _update_contrl_params(self):
        """
        Modify parameters based on the input YAML configuration for the ctrl experiment.

        Update configuration files (config.yaml, nuopc.runconfig etc),
        namelist and MOM_input for the control experiment if needed.
        """
        for file_name in os.listdir(self.base_path):
            if file_name.endswith('_in') or file_name.endswith('.nml'):  # Update parameters from namelists
                yaml_data = self.indata.get(file_name,None)
                if yaml_data:
                    if 'dynamics_nml' in yaml_data and 'turning_angle' in yaml_data['dynamics_nml']:
                        cosw = np.cos(yaml_data['dynamics_nml']['turning_angle'] * np.pi / 180.)
                        sinw = np.sin(yaml_data['dynamics_nml']['turning_angle'] * np.pi / 180.)
                        yaml_data['dynamics_nml']['cosw'] = cosw
                        yaml_data['dynamics_nml']['sinw'] = sinw
                        del yaml_data['dynamics_nml']['turning_angle']
                    nml_ctrl = f90nml.read(os.path.join(self.base_path,file_name))  # read existing namelist file from the control experiment
                    self._update_config_entries(nml_ctrl,yaml_data)  # update the namelist with the yaml input file
                    nml_ctrl.write(os.path.join(self.base_path,file_name),force=True)  # write the updated namelist back to the file

            if file_name in (('nuopc.runconfig', 'config.yaml')):  # Update config entries from `nuopc.runconfig` and `config_yaml`
                yaml_data = self.indata.get(file_name,None)
                if yaml_data:
                    tmp_file_path = os.path.join(self.base_path,file_name)
                    if file_name == 'nuopc.runconfig':
                        file_read = self.read_nuopc_config(tmp_file_path)
                        self._update_config_entries(file_read,yaml_data)
                        self.write_nuopc_config(file_read, tmp_file_path)
                    elif file_name == 'config.yaml':
                        file_read = self._read_ryaml(tmp_file_path)
                        self._update_config_entries(file_read,yaml_data)
                        self._write_ryaml(file_read, tmp_file_path)

            if file_name == 'MOM_input':  # Update parameters from `MOM_input`
                yaml_data = self.indata.get(file_name,None)
                if yaml_data:
                    MOM_inputParser = self._parser_mom6_input(os.path.join(self.base_path, file_name))  # parse existing MOM_input
                    param_dict = MOM_inputParser.param_dict  # read parameter dictionary
                    commt_dict = MOM_inputParser.commt_dict  # read comment dictionary
                    param_dict.update(yaml_data)
                    MOM_inputParser.writefile_MOM_input(os.path.join(self.base_path, file_name))  # overwrite to the same `MOM_input`

            if file_name == 'nuopc.runseq':  # Update only coupling timestep from `nuopc.runseq`
                yaml_data = self.indata.get('cpl_dt',None)
                if yaml_data:
                    nuopc_runseq_file = os.path.join(self.base_path,file_name)
                    self._update_cpl_dt_nuopc_seq(nuopc_runseq_file,yaml_data)

    def _check_and_commit_changes(self):
        """
        Checks the current state of the repo, stages relevant changes, and commits them.
        If no changes are detected, it provides a message indicating that no commit was made.
        """
        repo = git.Repo(self.base_path)
        print(f"Current base branch is: {repo.active_branch.name}")
        deleted_files = self._get_deleted_files(repo)
        if deleted_files:
            repo.index.remove(deleted_files,r=True)  # remove deleted files or `work` directory
        untracked_files = self._get_untracked_files(repo)
        changed_files = self._get_changed_files(repo)
        staged_files = set(untracked_files+changed_files)
        self._restore_swp_files(repo,staged_files)  # restore *.swp files in case users open any files during case is are running
        commit_message = f"Control experiment setup: Configure `{self.base_branch_name}` branch by `{self.yamlfile}`\n committed files/directories {staged_files}!"
        if staged_files:
            repo.index.add(staged_files)
            repo.index.commit(commit_message)
        else:
            print(f"Nothing changed, hence no further commits to the {self.base_path} repo!")

    def manage_perturb_expt(self):
        """
        Sets up perturbation experiments based on the configuration provided in `Expts_manager.yaml`.

        This function processes various parameter blocks defined in the YAML configuration, which may include
        1. namelist files (`_in`, `.nml`),
        2. MOM6 input files (`MOM_input`),
        3. `nuopc.runconfig`,
        4. `nuopc.runseq` (currently only for the coupling timestep).

        Raises:
            - Warning: If no namelist configurations are provided, the function issues a warning indicating that no parameter tuning tests will be conducted.
        """
        namelists = self.indata["namelists"]  # main section, top level key that groups different namlists
        if not namelists:
            warnings.warn("NO namelists were provided, hence there are no parameter-tunning tests!")
            return

        for k, nmls in namelists.items():
            if not nmls:
                continue
            self._process_params_blocks(k, nmls)

    def _process_params_blocks(self, k, nmls):
        """
        Determine the type of parameter block and processes it accordingly.
        
        Args:
            k (str): The key indicating the type of parameter block.
            nmls (dict): The namelist dictionary for the parameter block.
        """
        self.tag_model, expt_dir_name = self._determine_block_type(k)

        for k_sub in nmls:  # parameter groups, in which contains one or more specific parameters.
            self._process_params_group(k, k_sub, nmls, expt_dir_name, self.tag_model)

    def _determine_block_type(self, k):
        """
        Determine the type of parameter block based on the key.

        Args:
            k (str): The key indicating the type of parameter block.
        """
        if k.endswith(('_in', '.nml')):  # parameter blocks, in which contains one or more groups of parameters, e.g., input.nml, ice_in etc.
            tag_model = 'nml'
            expt_dir_name = k[:-3] if k.endswith('_in') else k[:-4]  # [Optional] user-defined directory name for each test
        elif k == 'MOM_input':
            tag_model = 'mom6'
            expt_dir_name = k
        elif k == 'nuopc.runseq':
            tag_model = 'cpl_dt'
            expt_dir_name = k[-6:]
        else:
            raise ValueError(f"Unsupported block type: {k}")
        return tag_model, expt_dir_name

    def _process_params_group(self, k, k_sub, nmls, expt_dir_name, tag_model):
        """
        Processes individual parameter groups based on the tag model.

        Args:
            k (str): The key indicating the type of parameter block.
            k_sub (str): The key for the specific parameter group.
            nmls (dict): The namelist dictionary for the parameter block.
            expt_dir_name (str): The user-defined directory name. [Optional]
            tag_model (str): The tag model indicating the type of parameter block.
        """
        if tag_model == 'nml':
            self._handle_nml_group(k, k_sub, expt_dir_name, nmls)
        elif tag_model == 'mom6':
            self._handle_mom6_group(k, k_sub, expt_dir_name, nmls)
        elif tag_model == 'cpl_dt':
            self._handle_cpl_dt_group(k, k_sub, expt_dir_name, nmls)
        self.previous_key = k_sub

    def _handle_nml_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `nml` tag model.
        """
        if k_sub.endswith('_nml') or k_sub.endswith(self.combo_suffix):
            self._process_parameter_group_common(k, k_sub, nmls, expt_dir_name)

    def _handle_mom6_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `mom6` tag model.
        """
        if k_sub.startswith(self.MOM_prefix):
            MOM_inputParser = self._parser_mom6_input(os.path.join(self.base_path, 'MOM_input'))
            commt_dict = MOM_inputParser.commt_dict
            self._process_parameter_group_common(k, k_sub, nmls, expt_dir_name, commt_dict=commt_dict)

    def _handle_cpl_dt_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `cpl_dt` tag model.
        """
        if k_sub.startswith(self.runseq_prefix):
            self._process_parameter_group_common(k, k_sub, nmls, expt_dir_name)

    def _process_parameter_group_common(self, k, k_sub, nmls, expt_dir_name, commt_dict=None):
        """
        Processes parameter groups common to all tag models.

        Args:
            k (str): The key indicating the type of parameter block.
            k_sub (str): The key for the specific parameter group.
            nmls (dict): The namelist dictionary for the parameter block.
            k_tmp_dir (str): The temporary directory name.
            commt_dict (dict, optional): A dictionary of comments, if applicable.
        """
        name_dict = nmls[k_sub]
        self._cal_num_expts(name_dict, k_sub)
        if self.previous_key and self.previous_key.startswith(expt_dir_name):
            self._valid_expt_names(nmls, name_dict)
        if k_sub.endswith(self.combo_suffix):
            self._generate_combined_dicts(name_dict,commt_dict,k_sub)
        else:
            self._generate_individual_dicts(name_dict,commt_dict,k_sub)
        self.setup_expts(k)
            
    def _cal_num_expts(self, name_dict, k_sub):
        """
        Evaluate the number of parameter-tunning experiments.
        """
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

    def _valid_expt_names(self, nmls, name_dict):
        """
        Compare the number of parameter-tunning experiments with [optional] user-defined experiment names
        """
        self.expt_names = nmls.get(self.previous_key)
        if self.expt_names and len(self.expt_names) != self.num_expts:
            raise ValueError(f"The number of user-defined experiment directories {self.expt_names} "
                             f"is different from that of tunning parameters {name_dict}!"
                             f"\nPlease double check the number or leave it/them blank!")

    def _generate_individual_dicts(self,name_dict,commt_dict,k_sub):
        """
        Each dictionary has a single key-value pair.
        """
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
        elif self.tag_model == 'nml':
            self.append_group_list = append_group_list

    def _generate_combined_dicts(self,name_dict,commt_dict,k_sub):
        """
        Generate a list of dictionaries where each dictionary contains all keys with values from the same index.
        """
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
        elif self.tag_model == 'nml':
            self.append_group_list = append_group_list

    def setup_expts(self,namelist_name):
        """ setup expts, and run expts"""
        for i in range(len(self.param_dict_change_list)):
            # for each experiment
            param_dict = self.param_dict_change_list[i]
            print(param_dict)

            if self.tag_model == 'nml':
                nml_group = self.append_group_list[i]

            if self.expt_names is None:
                expt_name = "_".join([f"{k}_{v}" for k,v in self.param_dict_change_list[i].items()])  # if `expt_names` does not exist, expt_name is set as the tunning parameters appending with associated values
            else:
                expt_name = self.expt_names[i]  # user-defined folder names for parameter-tunning experiments
            rel_path = os.path.join(self.test_path,expt_name)
            expt_path = os.path.join(self.dir_manager,rel_path)

            if os.path.exists(expt_path):
                print("-- not creating ", expt_path, " - already exists!")
            else:
                if self.check_skipping:
                    if self.tag_model == 'nml':
                        self._check_skipping(param_dict, nml_group, namelist_name, expt_path)

                print(f"Directory {expt_path} not exists, hence cloning template!")
                command = f"payu clone -B {self.base_branch_name} -b {self.branch_perturb} {self.base_path} {expt_path}"  # automatically leave a commit with expt uuid
                subprocess.run(command, shell=True, check=True)

            # Update `MOM_override` or/and `ice_in`
            if self.tag_model == 'mom6':
                # apply changes and write them to `MOM_override`
                MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, "MOM_override"))  # parse MOM_override
                MOM6_or_parser.param_dict, MOM6_or_parser.commt_dict = update_MOM6_params_override(param_dict,self.commt_dict_change)  # update the tunning parameters, values and associated comments
                MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))  # write to file

            elif self.tag_model == 'nml':
                # apply changes
                cice_path = os.path.join(expt_path,namelist_name)
                if nml_group.endswith(self.combo_suffix):  # rename the namlist by removing the suffix if the suffix with `_combo`
                    nml_group = nml_group[:-len(self.combo_suffix)]
                patch_dict = {nml_group: {}}
                for cice_name, cice_value in param_dict.items():
                    if cice_name == 'turning_angle':
                        cosw = np.cos(cice_value * np.pi / 180.)
                        sinw = np.sin(cice_value * np.pi / 180.)
                        patch_dict[nml_group]['cosw'] = cosw
                        patch_dict[nml_group]['sinw'] = sinw
                    else:  # for generic parameters
                        patch_dict[nml_group][cice_name] = cice_value
                f90nml.patch(cice_path, patch_dict, cice_path+'_tmp')
                os.rename(cice_path+'_tmp',cice_path)

            elif self.tag_model == 'cpl_dt':
                # apply changes
                nuopc_runseq_file = os.path.join(expt_path,"nuopc.runseq")
                self._update_cpl_dt_nuopc_seq(nuopc_runseq_file,param_dict[next(iter(param_dict.keys()))])

            if self.diag_pert:
                self._copy_diag_table(expt_path)

            if self.startfrom_str != 'rest':  # symlink restart directories
                link_restart = os.path.join('archive','restart'+self.startfrom_str)  # 
                restartpath = os.path.realpath(os.path.join(self.base_path,link_restart))  # restart dir from control experiment
                dest = os.path.join(expt_path, link_restart)  # restart dir symlink for each perturbation experiment
                # only create symlink if it doesnt exist or force_restart is enabled
                if not os.path.islink(dest) or self.force_restart or (os.path.islink(dest) and not os.path.exists(os.readlink(dest))):
                    if os.path.exists(dest) or os.path.islink(dest):
                        os.remove(dest)  # remove symlink
                    os.symlink(restartpath, dest)  # create symlink

            # optionally update nuopc_config for perturbation runs
            self._update_nuopc_config_perturb(expt_path)

            # Update config.yaml
            config_path = os.path.join(expt_path,"config.yaml")
            config_data = self._read_ryaml(config_path)
            config_data["jobname"] = expt_name
            self._write_ryaml(config_data, config_path)

            # Update metadata.yaml
            metadata_path = os.path.join(expt_path, "metadata.yaml")  # metadata path for each perturbation
            metadata = self._read_ryaml(metadata_path)  # load metadata of each perturbation
            if self.startfrom_str == 'rest':
                restartpath = 'rest'
            self._update_metadata_description(metadata,restartpath)  # update `description`
            self._remove_metadata_comments("description", metadata)  # remove None comments from `description`
            keywords = self._extract_metadata_keywords(param_dict)  # extract parameters from the change list
            metadata["keywords"] = f"{self.base_dir_name}, {self.branch_perturb}, {keywords}"  # update `keywords`
            self._remove_metadata_comments("keywords", metadata)  # remove None comments from `keywords`
            self._write_ryaml(metadata, metadata_path)  # write to file

            # clean `work` directory for failed jobs
            self._clean_workspace(expt_path)

            doneruns = len(glob.glob(os.path.join(expt_path,"archive","output[0-9][0-9][0-9]*")))
            # start runs, count existing runs and do additional runs if needed
            if self.nruns > 0:
                newruns = self.nruns - doneruns
                if newruns > 0:
                    command = f"cd {expt_path} && payu run -n {newruns} -f"
                    subprocess.run(command, check=False, shell=True)
                    print('\n')
                else:
                    print(f"-- `{expt_name}` has already completed {doneruns} runs! Hence stop running!\n")

        self.expt_names = None  # reset to None after the loop to update user-defined perturbation experiment names!

    def _check_skipping(self, param_dict, nml_group, namelist_name, expt_path):
        # create perturbation experiment - check if needs skipping!
        if self.tag_model == 'nml':
            if nml_group.endswith(self.combo_suffix):  # rename the namlist if suffix with `_combo`
                nml_group = nml_group[:-len(self.combo_suffix)]
            cice_name = param_dict.keys()
            if len(cice_name) == 1:  # one param:value pair
                cice_value = param_dict[list(cice_name)[0]]
            else:  # combination of param:value pairs
                cice_value = [param_dict[j] for j in cice_name]
    
            if 'turning_angle' in param_dict:
                cosw = np.cos(param_dict['turning_angle'] * np.pi / 180.)
                sinw = np.sin(param_dict['turning_angle'] * np.pi / 180.)
    
            # load nml of the control experiment
            self.nml_ctrl = f90nml.read(os.path.join(self.base_path,namelist_name))
    
            if all(cn in self.nml_ctrl.get(nml_group,{}) for cn in cice_name):  # cice_name (i.e. tunning parameter) may not be found in the control experiment
                if 'turning_angle' in param_dict:
                    skip = (self.nml_ctrl[nml_group]['cosw'] == cosw and
                            self.nml_ctrl[nml_group]['sinw'] == sinw and
                            all(self.nml_ctrl[nml_group].get(cn) == param_dict[cn] for cn in cice_name if cn not in ['cosw', 'sinw']))
                else:
                    skip = all(self.nml_ctrl[nml_group].get(cn) == param_dict[cn] for cn in cice_name)
            else:
                print(f"Not all {cice_name} are found in {nml_group}, hence not skipping!")
                skip = False
    
            if skip:
                print('-- not creating', expt_path, '- parameters are identical to the control experiment located at', self.base_path,'\n')
                return
    
        if self.tag_model == 'mom6': # might need MOM_parameter.all, because many parameters are in-default hence not shown up in `MOM_input` 
            #TODO
            pass

    def _clean_workspace(self,dir_path):
        work_dir = os.path.join(dir_path,'work')
        if os.path.islink(work_dir) and os.path.isdir(work_dir):  # in case any failed job
            # Payu sweep && setup to ensure the changes correctly && remove the `work` directory
            command = f"payu sweep && payu setup"
            subprocess.run(command, shell=True, check=False)
            print(f"Clean up a failed job {work_dir} and prepare it for resubmission.")

    def _parser_mom6_input(self, path):
        """ parse MOM6 input file """
        mom6parser = self.MOM6InputParser.MOM6InputParser()
        mom6parser.read_input(path)
        mom6parser.parse_lines()
        return mom6parser

    def _update_nuopc_config_perturb(self, path):
        """ Update nuopc.runconfig for the ctrl run """
        nuopc_input = self.indata.get("perturb_run_config",None)
        if nuopc_input is not None:
            nuopc_file_path = os.path.join(path,"nuopc.runconfig")
            nuopc_runconfig = self.read_nuopc_config(nuopc_file_path)
            self._update_config_entries(nuopc_runconfig,nuopc_input)
            self.write_nuopc_config(nuopc_runconfig, nuopc_file_path)

    def _update_cpl_dt(self,path):
        """ Update coupling timestep through nuopc.runseq for the ctrl run """
        cpl_dt_input = self.indata.get("cpl_dt",None)
        if cpl_dt_input is not None:
            nuopc_runseq_file = os.path.join(path,"nuopc.runseq")
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

    def _write_ryaml(self,data,yaml_path):
        """ Write yaml file and preserve comments"""
        with open(yaml_path, "w") as f:
            ryaml.dump(data,f)

    def _update_metadata_description(self, metadata,restartpath):
        """Update metadata description with experiment details."""
        tmp_string1 = (f"\nNOTE: this is a perturbation experiment, but the description above is for the control run."
                    f"\nThis perturbation experiment is based on the control run {self.base_path} from {self.base_branch_name}")
        tmp_string2 = f"\nbut with initial condition {restartpath}."
        desc = metadata["description"]
        if desc is None:
            desc = ""
        if tmp_string1.strip() not in desc.strip():
            desc += tmp_string1
        if tmp_string2.strip() not in desc.strip():
            desc += tmp_string2
        metadata["description"] = LiteralString(desc)

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

        yamlfile = os.path.join(self.dir_manager,INPUT_YAML)
        self.load_variables(yamlfile)
        self.load_tools()
        self.create_test_path()
        self.manage_ctrl_expt()
        if self.run_namelists:
            print("==== Start perturbation experiments ====")
            self.manage_perturb_expt()
            
        
if __name__ == "__main__":
    expt_manager = Expts_manager()
    expt_manager.main()