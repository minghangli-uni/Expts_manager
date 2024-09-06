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
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


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
    override_param_dict_change = {
        f"#override {k}": v for k, v in param_dict_change.items()
    }
    override_commt_dict_change = {
        f"#override {k}": v for k, v in commt_dict_change.items()
    }
    return override_param_dict_change, override_commt_dict_change


class Expts_manager(object):
    """
    A class to manage ACCESS-OM3 experiment runs, including control and perturbation experiments.
    Attributes:
        MOM_prefix (str): Prefix for MOM6 parameters.
        nml_suffix (str): Suffix for namelist parameters.
        runseq_prefix (str): Prefix for the coupling timestep in `nuopc.runseq`.
        combo_suffix (str): Suffix for combo perturbation experiments, i.e., multiple-parameter tests.
        branch_perturb (str): branch name for the perturbation.
    """

    DIR_MANAGER = os.getcwd()

    def __init__(
        self,
        MOM_prefix: str = "MOM_list",
        nml_suffix: str = "_nml",
        runseq_prefix: str = "runseq_list",
        combo_suffix: str = "_combo",
        branch_perturb: str = "perturb",
    ):

        self.dir_manager = self.DIR_MANAGER
        self.MOM_prefix = MOM_prefix
        self.nml_suffix = nml_suffix
        self.runseq_prefix = runseq_prefix
        self.branch_perturb = branch_perturb
        self.combo_suffix = combo_suffix

    def load_variables(self, yamlfile):
        """
        Loads variables from the input yaml file
        Args:
            yamlfile (str): Path to the YAML configuration file, i.e., Expts_manager.yaml.
        Attributes:
            yamlfile (str): Path to the YAML configuration file.
            indata (dict): Data loaded from the YAML file.
            utils_url (str): Git url for the om3-utils tool.
            utils_branch_name (str): Branch name for the om3-utils tool.
            utils_dir_name (str): User-defined directory for the om3-utils tool.
            base_url (dict): Git url for the ACCESS-OM3 configuration.
            base_commit (str): Specific git commit for the ACCESS-OM3 configuration.
            base_dir_name (str): User-defined directory name for the baseline control experiment.
            base_branch_name (str): User-defined branch name for the control experiment.
            test_path (str): User-defined path for test runs, including control and perturbation experiments.
            startfrom (int/str): Restart number of the control experiment used as an initial condition for perturbation tests; use 'rest' to start from the initial state.
            startfrom_str (str): String representation of `startfrom`, padded to three digits.
            ctrl_nruns (int): Number of control runs. It is associated with total number of output directories that have been generated.
            pert_nruns (int): Number of perturbation experiment runs; associated with total number of output directories that have been generated.
        """
        self.yamlfile = yamlfile
        self.indata = self._read_ryaml(yamlfile)

        self.utils_url = self.indata["utils_url"]
        self.utils_dir_name = self.indata["utils_dir_name"]
        self.utils_branch_name = self.indata["utils_branch_name"]

        self.base_url = self.indata["base_url"]
        self.base_commit = self.indata["base_commit"]
        self.base_dir_name = self.indata["base_dir_name"]
        self.base_branch_name = self.indata["base_branch_name"]

        self.test_path = self.indata["test_path"]

        self.diag_url = self.indata.get("diag_url", None)
        self.diag_branch_name = self.indata.get("diag_branch_name", None)
        self.diag_dir_name = self.indata.get("diag_dir_name", None)
        self.diag_ctrl = self.indata.get("diag_ctrl", False)
        self.diag_pert = self.indata.get("diag_pert", False)

        self.ctrl_nruns = self.indata.get("ctrl_nruns", 0)
        self.run_namelists = self.indata.get("run_namelists", False)
        self.check_duplicate_jobs = self.indata.get("check_duplicate_jobs", True)
        self.check_skipping = self.indata.get("check_skipping", False)
        self.force_restart = self.indata.get("force_restart", False)
        self.startfrom = self.indata["startfrom"]
        self.startfrom_str = str(self.startfrom).strip().lower().zfill(3)
        self.nruns = self.indata.get("nruns", 0)

        self._initialise_variables()

    def _initialise_variables(self):
        """
        Initialises variables from experiment setups
        nml_ctrl (f90nml): f90 namlist for the interested parameters. It is used as a base to modify for perturbation experiments.
        tag_model (str): Switch for tuning parameters between f90 namelist and MOM_input.
        param_dict_change_list list[dict]: Specific for MOM_input, the list containing tunning parameter dictionaries.
        commt_dict_change (dict): Specific for MOM_input, dictionary of comments for parameters.
        append_group_list (list): Specific for f90nml, the list containing tunning parameters.
        expt_names list(str): Optional user-defined directory names for perturbation experiments.
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
        Loads external tools required for the experiments.
        """

        # currently import from a fork: https://github.com/minghangli-uni/om3-utils
        # will update the tool when it is merged to COSIMA/om3-utils
        def _clone_repo(branch_name, url, path, tool_name):
            if not os.path.isdir(path):
                command = (
                    f"git clone --branch {branch_name} {url} {path} --single-branch"
                )
                subprocess.run(command, shell=True, check=True)
                print(f"Cloning {tool_name} for use!")
            else:
                print(f"{path} already exists!")

        # om3-utils is a must
        utils_path = os.path.join(self.dir_manager, self.utils_dir_name)
        _clone_repo(
            self.utils_branch_name, self.utils_url, utils_path, self.utils_dir_name
        )

        # make_diag_table is [optional]
        self.diag_path = (
            os.path.join(self.dir_manager, self.diag_dir_name)
            if self.diag_dir_name
            else None
        )
        _clone_repo(
            self.diag_branch_name, self.diag_url, self.diag_path, self.diag_dir_name
        )

        sys.path.extend([utils_path, self.diag_path])

        # load modules from om3-utils
        from om3utils import MOM6InputParser
        from om3utils.nuopc_config import read_nuopc_config, write_nuopc_config

        self.MOM6InputParser = MOM6InputParser
        self.read_nuopc_config = read_nuopc_config
        self.write_nuopc_config = write_nuopc_config

    def create_test_path(self):
        """
        Creates the local test directory for blocks of parameter testing.
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
        self.base_path = os.path.join(
            self.dir_manager, self.test_path, self.base_dir_name
        )
        base_path = self.base_path
        ctrl_nruns = self.ctrl_nruns

        if os.path.exists(base_path):
            print(f"Base path is already created and located at {base_path}")
            if self._count_file_nums() == 4:
                print(
                    "previous commit fails, please try with an updated commit hash for the control experiment!"
                )
                # extract specific configuration via commit hash
                self._extract_config_via_commit()

                # [optional] modify diag_table
                if self.diag_ctrl and self.diag_path:
                    self._copy_diag_table(base_path)

                # setup the control experiments
                self._setup_ctrl_expt()

                # Checks the current state of the repo, commits relevant changes.
                self._check_and_commit_changes()
        else:
            # clone the template repo and setup the control branch
            self._clone_template_repo()

            # extract specific configuration via commit hash
            self._extract_config_via_commit()

            # [optional] modify diag_table
            if self.diag_ctrl and self.diag_path:
                self._copy_diag_table(base_path)

            # setup the control experiments
            self._setup_ctrl_expt()

            # Checks the current state of the repo, commits relevant changes.
            self._check_and_commit_changes()

        # check exisiting pbs jobs
        pbs_jobs = self._output_existing_pbs_jobs()

        # check duplicated running jobs
        if self.check_duplicate_jobs:
            duplicated_bool = self._check_duplicated_jobs(pbs_jobs, base_path)
        else:
            duplicated_bool = False

        # start control runs, count existing runs and do additional runs if needed
        self._start_experiment_runs(
            base_path, self.base_dir_name, duplicated_bool, ctrl_nruns
        )

    def _clone_template_repo(self):
        """
        Clones the template repo.
        """
        print(f"Cloning template from {self.base_url} to {self.base_path}")
        command = f"payu clone {self.base_url} {self.base_path}"
        subprocess.run(command, shell=True, check=False)

    def _extract_config_via_commit(self):
        """
        Extract specific configuration via commit hash.
        """
        templaterepo = git.Repo(self.base_path)
        print(
            f"Check out commit {self.base_commit} and creat new branch {self.base_branch_name}!"
        )
        # checkout the new branch from the specific template commit
        templaterepo.git.checkout("-b", self.base_branch_name, self.base_commit)

    def _copy_diag_table(self, path):
        """
        Copies the diagnostic table (`diag_table`) to the specified path if a path is defined.
        """
        if self.diag_path:
            command = f"scp {os.path.join(self.diag_path,'diag_table')} {path}"
            subprocess.run(command, shell=True, check=False)
            print(f"Copy diag_table to {path}")
        else:
            print(
                f"{self.diag_path} is not defined, hence skip copy diag_table to the control experiment"
            )

    def _count_file_nums(self):
        """
        Counts the number of file numbers.
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
        Modifies parameters based on the input YAML configuration for the ctrl experiment.

        Updates configuration files (config.yaml, nuopc.runconfig etc),
        namelist and MOM_input for the control experiment if needed.
        """
        for file_name in os.listdir(self.base_path):
            # Update parameters from namelists
            if file_name.endswith("_in") or file_name.endswith(".nml"):
                yaml_data = self.indata.get(file_name, None)
                if yaml_data:
                    if (
                        "dynamics_nml" in yaml_data
                        and "turning_angle" in yaml_data["dynamics_nml"]
                    ):
                        cosw = np.cos(
                            yaml_data["dynamics_nml"]["turning_angle"] * np.pi / 180.0
                        )
                        sinw = np.sin(
                            yaml_data["dynamics_nml"]["turning_angle"] * np.pi / 180.0
                        )
                        yaml_data["dynamics_nml"]["cosw"] = cosw
                        yaml_data["dynamics_nml"]["sinw"] = sinw
                        del yaml_data["dynamics_nml"]["turning_angle"]

                    # read existing namelist file from the control experiment
                    nml_ctrl = f90nml.read(os.path.join(self.base_path, file_name))
                    # update the namelist with the YAML input file
                    self._update_config_entries(nml_ctrl, yaml_data)
                    # write the updated namelist back to the file
                    nml_ctrl.write(os.path.join(self.base_path, file_name), force=True)

            # Update config entries from `nuopc.runconfig` and `config_yaml`
            if file_name in (("nuopc.runconfig", "config.yaml")):
                yaml_data = self.indata.get(file_name, None)
                if yaml_data:
                    tmp_file_path = os.path.join(self.base_path, file_name)
                    if file_name == "nuopc.runconfig":
                        file_read = self.read_nuopc_config(tmp_file_path)
                        self._update_config_entries(file_read, yaml_data)
                        self.write_nuopc_config(file_read, tmp_file_path)
                    elif file_name == "config.yaml":
                        file_read = self._read_ryaml(tmp_file_path)
                        yaml_data["jobname"] = self.base_dir_name
                        if yaml_data["jobname"] != self.base_dir_name:
                            raise ValueError(
                                f"jobname must be the same as {self.base_dir_name}!"
                            )
                        self._update_config_entries(file_read, yaml_data)
                        self._write_ryaml(file_read, tmp_file_path)

            # Update parameters from `MOM_input`
            if file_name == "MOM_input":
                yaml_data = self.indata.get(file_name, None)
                if yaml_data:
                    # parse existing MOM_input
                    MOM_inputParser = self._parser_mom6_input(
                        os.path.join(self.base_path, file_name)
                    )
                    param_dict = MOM_inputParser.param_dict  # read parameter dictionary
                    commt_dict = MOM_inputParser.commt_dict  # read comment dictionary
                    param_dict.update(yaml_data)
                    # overwrite to the same `MOM_input`
                    MOM_inputParser.writefile_MOM_input(
                        os.path.join(self.base_path, file_name)
                    )

            # Update only coupling timestep from `nuopc.runseq`
            if file_name == "nuopc.runseq":
                yaml_data = self.indata.get("cpl_dt", None)
                if yaml_data:
                    nuopc_runseq_file = os.path.join(self.base_path, file_name)
                    self._update_cpl_dt_nuopc_seq(nuopc_runseq_file, yaml_data)

    def _check_and_commit_changes(self):
        """
        Checks the current state of the repo, stages relevant changes, and commits them.
        If no changes are detected, it provides a message indicating that no commit was made.
        """
        repo = git.Repo(self.base_path)
        print(f"Current base branch is: {repo.active_branch.name}")
        deleted_files = self._get_deleted_files(repo)
        # remove deleted files or `work` directory
        if deleted_files:
            repo.index.remove(deleted_files, r=True)
        untracked_files = self._get_untracked_files(repo)
        changed_files = self._get_changed_files(repo)
        staged_files = set(untracked_files + changed_files)
        # restore *.swp files in case users open any files during case is are running
        self._restore_swp_files(repo, staged_files)
        commit_message = f"Control experiment setup: Configure `{self.base_branch_name}` branch by `{self.yamlfile}`\n committed files/directories {staged_files}!"
        if staged_files:
            repo.index.add(staged_files)
            repo.index.commit(commit_message)
        else:
            print(
                f"Nothing changed, hence no further commits to the {self.base_path} repo!"
            )

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
        namelists = self.indata[
            "namelists"
        ]  # main section, top level key that groups different namlists
        if not namelists:
            warnings.warn(
                "NO namelists were provided, hence there are no parameter-tunning tests!"
            )
            return

        for k, nmls in namelists.items():
            if not nmls:
                continue
            self._process_params_blocks(k, nmls)

    def _process_params_blocks(self, k, nmls):
        """
        Determines the type of parameter block and processes it accordingly.

        Args:
            k (str): The key indicating the type of parameter block.
            nmls (dict): The namelist dictionary for the parameter block.
        """
        self.tag_model, expt_dir_name = self._determine_block_type(k)

        # parameter groups, in which contains one or more specific parameters
        for k_sub in nmls:
            self._process_params_group(k, k_sub, nmls, expt_dir_name, self.tag_model)

    def _determine_block_type(self, k):
        """
        Determines the type of parameter block based on the key.

        Args:
            k (str): The key indicating the type of parameter block.
        """
        # parameter blocks, in which contains one or more groups of parameters, e.g., input.nml, ice_in etc.
        if k.endswith(("_in", ".nml")):
            tag_model = "nml"
            # [Optional] The key in the YAML file specifies a list of user-defined directory names related to parameter testing.
            expt_dir_name = k[:-3] if k.endswith("_in") else k[:-4]
        elif k == "MOM_input":
            tag_model = "mom6"
            expt_dir_name = k
        elif k == "nuopc.runseq":
            tag_model = "cpl_dt"
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
            expt_dir_name (str, optional): The key in the YAML file specifies a list of user-defined directory names related to parameter testing.
            tag_model (str): The tag model indicating the type of parameter block.
        """
        if tag_model == "nml":
            self._handle_nml_group(k, k_sub, expt_dir_name, nmls)
        elif tag_model == "mom6":
            self._handle_mom6_group(k, k_sub, expt_dir_name, nmls)
        elif tag_model == "cpl_dt":
            self._handle_cpl_dt_group(k, k_sub, expt_dir_name, nmls)
        self.previous_key = k_sub

    def _handle_nml_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `nml` tag model.
        """
        if k_sub.endswith(self.nml_suffix) or k_sub.endswith(self.combo_suffix):
            self._process_parameter_group_common(k, k_sub, nmls, expt_dir_name)

    def _handle_mom6_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `mom6` tag model.
        """
        if k_sub.startswith(self.MOM_prefix):
            MOM_inputParser = self._parser_mom6_input(
                os.path.join(self.base_path, "MOM_input")
            )
            commt_dict = MOM_inputParser.commt_dict
            self._process_parameter_group_common(
                k, k_sub, nmls, expt_dir_name, commt_dict=commt_dict
            )

    def _handle_cpl_dt_group(self, k, k_sub, expt_dir_name, nmls):
        """
        Handles namelist parameter groups specific to `cpl_dt` tag model.
        """
        if k_sub.startswith(self.runseq_prefix):
            self._process_parameter_group_common(k, k_sub, nmls, expt_dir_name)

    def _process_parameter_group_common(
        self, k, k_sub, nmls, expt_dir_name, commt_dict=None
    ):
        """
        Processes parameter groups to all tag models.

        Args:
            k (str): The key indicating the type of parameter block.
            k_sub (str): The key for the specific parameter group.
            nmls (dict): The namelist dictionary for the parameter block.
            expt_dir_name (str, optional): The key in the YAML file specifies a list of user-defined directory names related to parameter testing.
            commt_dict (dict, optional): A dictionary of comments, if applicable.
        """
        name_dict = nmls[k_sub]
        self._cal_num_expts(name_dict, k_sub)
        if self.previous_key and self.previous_key.startswith(expt_dir_name):
            self._valid_expt_names(nmls, name_dict)
        if k_sub.endswith(self.combo_suffix):
            self._generate_combined_dicts(name_dict, commt_dict, k_sub)
        else:
            self._generate_individual_dicts(name_dict, commt_dict, k_sub)
        self.setup_expts(k)

    def _cal_num_expts(self, name_dict, k_sub):
        """
        Evaluates the number of parameter-tunning experiments.
        """
        if k_sub.endswith(self.combo_suffix):
            if isinstance(next(iter(name_dict.values())), list):
                self.num_expts = len(next(iter(name_dict.values())))
            else:
                self.num_expts = 1
        else:
            self.num_expts = 0
            for v_s in name_dict.values():
                if isinstance(v_s, list):
                    self.num_expts += len(v_s)
                else:
                    self.num_expts = 1

    def _valid_expt_names(self, nmls, name_dict):
        """
        Compares the number of parameter-tunning experiments with [optional] user-defined experiment names.
        """
        self.expt_names = nmls.get(self.previous_key)
        if self.expt_names and len(self.expt_names) != self.num_expts:
            raise ValueError(
                f"The number of user-defined experiment directories {self.expt_names} "
                f"is different from that of tunning parameters {name_dict}!"
                f"\nPlease double check the number or leave it/them blank!"
            )

    def _generate_individual_dicts(self, name_dict, commt_dict, k_sub):
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
        if self.tag_model == "mom6":
            self.commt_dict_change = {k: commt_dict.get(k, "") for k in name_dict}
        elif self.tag_model == "nml":
            self.append_group_list = append_group_list

    def _generate_combined_dicts(self, name_dict, commt_dict, k_sub):
        """
        Generates a list of dictionaries where each dictionary contains all keys with values from the same index.
        """
        param_dict_change_list = []
        append_group_list = []
        for i in range(self.num_expts):
            param_dict_change = {k: name_dict[k][i] for k in name_dict}
            append_group = k_sub
            append_group_list.append(append_group)
            param_dict_change_list.append(param_dict_change)
        self.param_dict_change_list = param_dict_change_list
        if self.tag_model == "mom6":
            self.commt_dict_change = {k: commt_dict.get(k, "") for k in name_dict}
        elif self.tag_model == "nml":
            self.append_group_list = append_group_list

    def setup_expts(self, parameter_block):
        """
        Sets up perturbation experiments based on the YAML input file provided in `Expts_manager.yaml`.
        """
        for i, param_dict in enumerate(self.param_dict_change_list):
            print(param_dict)

            # generate perturbation experiment directory names
            expt_name = self._generate_expt_names(i)

            # perturbation experiment path
            expt_path = os.path.join(self.dir_manager, self.test_path, expt_name)

            # generate perturbation experiment directory
            if os.path.exists(expt_path):
                print(f"-- not creating {expt_path} - already exists!")
            else:
                self._generate_expt_directory(expt_path, parameter_block, i)

            # update params for each parameter block
            if self.tag_model == "mom6":
                self._update_mom6_params(expt_path, param_dict)
            elif self.tag_model == "nml":
                self._update_nml_params(expt_path, param_dict, parameter_block, i)
            elif self.tag_model == "cpl_dt":
                self._update_cpl_dt_params(expt_path, param_dict)

            # optionally update diag_table for perturbation runs
            if self.diag_pert and self.diag_path:
                self._copy_diag_table(expt_path)

            # symlink restart directories
            if self.startfrom_str != "rest":
                self._generate_restart_symlink(expt_path)

            # optionally update nuopc.runconfig for perturbation runs
            self._update_nuopc_config_perturb(expt_path)

            self._update_config_yaml_perturb(expt_path, expt_name)
            self._update_metadata_yaml_perturb(expt_path, param_dict)

            # clean `work` directory for failed jobs
            self._clean_workspace(expt_path)

            # check exisiting pbs jobs
            pbs_jobs = self._output_existing_pbs_jobs()

            # check duplicated running jobs
            if self.check_duplicate_jobs:
                duplicated_bool = self._check_duplicated_jobs(pbs_jobs, expt_path)
            else:
                duplicated_bool = False

            # start runs, count existing runs and do additional runs if needed
            self._start_experiment_runs(
                expt_path, expt_name, duplicated_bool, self.nruns
            )

        self.expt_names = None  # reset to None after the loop to update user-defined perturbation experiment names!

    def _generate_expt_names(self, indx):
        if self.expt_names is None:
            # if `expt_names` does not exist, `expt_names` is set as the tunning parameters appending with associated values.
            return "_".join(
                [f"{k}_{v}" for k, v in self.param_dict_change_list[indx].items()]
            )
        # user-defined directory names for each parameter-tunning experiment.
        return self.expt_names[indx]

    def _generate_expt_directory(self, expt_path, parameter_block, indx):
        """
        Generates a new experiment directory by cloning the control experiment.
        Checks if the tuning parameter matches the control experiment,
        this validation currently applies only to `nml` files.

        Args:
            expt_path (str): The path to the experiment directory.
            param_dict (dict): The dictionary of parameters to update.
        """
        if self.check_skipping:
            if self.tag_model == "nml":
                self._check_skipping(
                    self.param_dict_change_list[indx],
                    self.append_group_list[indx],
                    parameter_block,
                    expt_path,
                )
            elif self.tag_model == "mom6":  # TODO
                pass
            elif self.tag_model == "cpl_dt":  # TODO
                pass

        print(f"Directory {expt_path} not exists, hence cloning template!")
        command = f"payu clone -B {self.base_branch_name} -b {self.branch_perturb} {self.base_path} {expt_path}"  # automatically leave a commit with expt uuid
        subprocess.run(command, shell=True, check=True)

    def _update_mom6_params(self, expt_path, param_dict):
        """
        Updates MOM6 parameters in the 'MOM_override' file.

        Args:
            expt_path (str): The path to the experiment directory.
            param_dict (dict): The dictionary of parameters to update.
        """
        MOM6_or_parser = self._parser_mom6_input(
            os.path.join(expt_path, "MOM_override")
        )
        MOM6_or_parser.param_dict, MOM6_or_parser.commt_dict = (
            update_MOM6_params_override(param_dict, self.commt_dict_change)
        )
        MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path, "MOM_override"))

    def _update_nml_params(self, expt_path, param_dict, parameter_block, indx):
        """
        Updates namelist parameters and overwrites namelist file.

        Args:
            expt_path (str): The path to the experiment directory.
            param_dict (dict): The dictionary of parameters to update.
            parameter_block (str): The name of the namelist file.
        """

        nml_path = os.path.join(expt_path, parameter_block)
        nml_group = self.append_group_list[indx]

        if nml_group.endswith(self.combo_suffix):
            nml_group = nml_group[
                : -len(self.combo_suffix)
            ]  # rename the namlist by removing the suffix if the suffix with `_combo`

        patch_dict = {nml_group: {}}
        for nml_name, nml_value in param_dict.items():
            if nml_name == "turning_angle":
                cosw = np.cos(nml_value * np.pi / 180.0)
                sinw = np.sin(nml_value * np.pi / 180.0)
                patch_dict[nml_group]["cosw"] = cosw
                patch_dict[nml_group]["sinw"] = sinw
            else:  # for generic parameters
                patch_dict[nml_group][nml_name] = nml_value

        f90nml.patch(nml_path, patch_dict, nml_path + "_tmp")
        os.rename(nml_path + "_tmp", nml_path)

    def _update_cpl_dt_params(self, expt_path, param_dict):
        """
        Updates coupling timestep parameters.

        Args:
            expt_path (str): The path to the experiment directory.
            param_dict (dict): The dictionary of parameters to update.
        """
        nuopc_runseq_file = os.path.join(expt_path, "nuopc.runseq")
        self._update_cpl_dt_nuopc_seq(
            nuopc_runseq_file, param_dict[next(iter(param_dict.keys()))]
        )

    def _generate_restart_symlink(self, expt_path):
        """
        Generates a symlink to the restart directory if needed.

        Args:
            expt_path (str): The path to the experiment directory.
        """
        link_restart = os.path.join("archive", "restart" + self.startfrom_str)
        # restart dir from control experiment
        restartpath = os.path.realpath(os.path.join(self.base_path, link_restart))
        # restart dir symlink for each perturbation experiment
        dest = os.path.join(expt_path, link_restart)

        # only generate symlink if it doesnt exist or force_restart is enabled
        if (
            not os.path.islink(dest)
            or self.force_restart
            or (os.path.islink(dest) and not os.path.exists(os.readlink(dest)))
        ):
            if os.path.exists(dest) or os.path.islink(dest):
                os.remove(dest)  # remove symlink
            os.symlink(restartpath, dest)  # generate symlink

    def _update_nuopc_config_perturb(self, path):
        """
        Updates nuopc.runconfig for perturbation experiment runs.
        """
        nuopc_input = self.indata.get("perturb_run_config", None)
        if nuopc_input is not None:
            nuopc_file_path = os.path.join(path, "nuopc.runconfig")
            nuopc_runconfig = self.read_nuopc_config(nuopc_file_path)
            self._update_config_entries(nuopc_runconfig, nuopc_input)
            self.write_nuopc_config(nuopc_runconfig, nuopc_file_path)

    def _update_config_yaml_perturb(self, expt_path, expt_name):
        """
        Updates `jobname` only for now.

        Args:
            expt_path (str): The path to the perturbation experiment directory.
            expt_name (str): The name of the perturbation experiment.
        """
        config_path = os.path.join(expt_path, "config.yaml")
        config_data = self._read_ryaml(config_path)
        config_data["jobname"] = expt_name
        self._write_ryaml(config_data, config_path)

    def _update_metadata_yaml_perturb(self, expt_path, param_dict):
        """
        Updates the `metadata.yaml` file with relevant metadata.

        Args:
            expt_path (str): The path to the perturbation experiment directory.
            param_dict (dict): The dictionary of parameters to include in metadata.
        """
        metadata_path = os.path.join(expt_path, "metadata.yaml")
        metadata = self._read_ryaml(metadata_path)  # load metadata of each perturbation
        if self.startfrom_str == "rest":
            restartpath = "rest"
        self._update_metadata_description(metadata, restartpath)  # update `description`
        self._remove_metadata_comments(
            "description", metadata
        )  # remove None comments from `description`
        keywords = self._extract_metadata_keywords(
            param_dict
        )  # extract parameters from the change list
        metadata["keywords"] = (
            f"{self.base_dir_name}, {self.branch_perturb}, {keywords}"  # update `keywords`
        )
        self._remove_metadata_comments(
            "keywords", metadata
        )  # remove None comments from `keywords`
        self._write_ryaml(metadata, metadata_path)  # write to file

    def _clean_workspace(self, dir_path):
        """
        Cleans `work` directory for failed jobs.
        """
        work_dir = os.path.join(dir_path, "work")
        # in case any failed job
        if os.path.islink(work_dir) and os.path.isdir(work_dir):
            # Payu sweep && setup to ensure the changes correctly && remove the `work` directory
            command = f"payu sweep && payu setup"
            subprocess.run(command, shell=True, check=False)
            print(f"Clean up a failed job {work_dir} and prepare it for resubmission.")

    def _output_existing_pbs_jobs(self):
        """
        Checks the existing qstat pbs information.
        """
        current_job_status_path = os.path.join(self.dir_manager, "current_job_status")
        command = f"qstat -f > {current_job_status_path}"
        subprocess.run(command, shell=True, check=False)

        pbs_jobs = {}
        current_key = None
        current_value = ""
        job_id = None
        with open(current_job_status_path, "r") as f:
            pbs_job_file = f.read()

        pbs_job_file = pbs_job_file.replace("\t", "        ")

        for line in pbs_job_file.splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("Job Id:"):
                job_id = line.split(":", 1)[1].strip()
                pbs_jobs[job_id] = {}
                current_key = None
                current_value = ""
            elif line.startswith("        ") and current_key:  # 8 indents multi-line
                current_value += line.strip()
            elif line.startswith("    ") and " = " in line:  # 4 indents for new pair
                # Save the previous multi-line value
                if current_key:
                    pbs_jobs[job_id][current_key] = current_value.strip()
                key, value = line.split(" = ", 1)  # save key
                current_key = key.strip()
                current_value = value.strip()
        return pbs_jobs

    def _check_duplicated_jobs(self, pbs_jobs, expt_path):

        def extract_current_and_parent_folder(tmp_path):

            # extract base_name or expt_name from pbs jobs
            folder_path = "/" + "/".join(tmp_path.split("/")[1:-1])

            # extract test_path from pbs jobs
            parent_path = "/" + "/".join(tmp_path.split("/")[1:-2])

            return folder_path, parent_path

        parent_paths = {}
        for job_id, job_info in pbs_jobs.items():
            folder_path, parent_path = extract_current_and_parent_folder(
                job_info["Error_Path"]
            )
            if parent_path not in parent_paths:
                parent_paths[parent_path] = []
            parent_paths[parent_path].append(folder_path)

        duplicated = False

        for parent_path, folder_paths in parent_paths.items():
            if expt_path in folder_paths:
                print(
                    f"You have duplicated runs for folder '{os.path.basename(expt_path)}' in the same folder '{parent_path}', "
                    f"hence not submitting this job!\n"
                )
                duplicated = True
        return duplicated

    def _start_experiment_runs(self, expt_path, expt_name, duplicated, num_runs):
        """
        Runs perturbation experiments.

        Args:
            expt_path (str): The path to the control/perturbation experiment directory.
            expt_name (str): The name of the control/perturbation experiment.
        """

        def runs():
            doneruns = len(
                glob.glob(os.path.join(expt_path, "archive", "output[0-9][0-9][0-9]*"))
            )
            newruns = num_runs - doneruns
            if newruns > 0:
                print(f"\nRun experiment -n {newruns}\n")
                command = f"cd {expt_path} && payu run -n {newruns} -f"
                subprocess.run(command, shell=True, check=False)
                print("\n")
            else:
                print(
                    f"-- `{expt_name}` has already completed {doneruns} runs! Hence, stopping further runs.\n"
                )

        if not duplicated:
            if num_runs > 0:
                runs()
            else:
                print(
                    f"-- number of runs is {num_runs}, hence no new experiments will start!\n"
                )

    def _start_experiment_runs2(self, expt_path, expt_name, pbs_jobs, num_runs):
        print(expt_name)
        """
        Runs perturbation experiments.

        Args:
            expt_path (str): The path to the control/perturbation experiment directory.
            expt_name (str): The name of the control/perturbation experiment.
        """

        def runs():
            doneruns = len(
                glob.glob(os.path.join(expt_path, "archive", "output[0-9][0-9][0-9]*"))
            )
            newruns = num_runs - doneruns
            if newruns > 0:
                print(f"\nRun experiment -n {newruns}\n")
                command = f"cd {expt_path} && payu run -n {newruns} -f"
                subprocess.run(command, shell=True, check=False)
                print("\n")
            else:
                print(
                    f"-- `{expt_name}` has already completed {doneruns} runs! Hence, stopping further runs.\n"
                )

        if pbs_jobs and any(
            job_info["Error_Path"].split("/")[-2] == expt_name
            for job_info in pbs_jobs.values()
        ):
            print(
                f"-- `{expt_name}` is running now and hence skip rusubmitting the pbs job! \n"
            )
        else:
            if num_runs > 0:
                runs()
            else:
                print(
                    f"-- number of runs is {num_runs}, hence no new experiments will start!\n"
                )

    def _check_skipping(self, param_dict, nml_group, parameter_block, expt_path):
        """
        Checks if the tuning parameter matches the control experiment,
        this validation currently applies only to `nml` files.
        """
        if self.tag_model == "nml":
            # rename the namlist if suffix with `_combo`
            if nml_group.endswith(self.combo_suffix):
                nml_group = nml_group[: -len(self.combo_suffix)]
            nml_name = param_dict.keys()
            if len(nml_name) == 1:  # one param:value pair
                nml_value = param_dict[list(nml_name)[0]]
            else:  # combination of param:value pairs
                nml_value = [param_dict[j] for j in nml_name]

            if "turning_angle" in param_dict:
                cosw = np.cos(param_dict["turning_angle"] * np.pi / 180.0)
                sinw = np.sin(param_dict["turning_angle"] * np.pi / 180.0)

            # load nml of the control experiment
            self.nml_ctrl = f90nml.read(os.path.join(self.base_path, parameter_block))

            if all(
                cn in self.nml_ctrl.get(nml_group, {}) for cn in nml_name
            ):  # nml_name (i.e. tunning parameter) may not be found in the control experiment
                if "turning_angle" in param_dict:
                    skip = (
                        self.nml_ctrl[nml_group]["cosw"] == cosw
                        and self.nml_ctrl[nml_group]["sinw"] == sinw
                        and all(
                            self.nml_ctrl[nml_group].get(cn) == param_dict[cn]
                            for cn in nml_name
                            if cn not in ["cosw", "sinw"]
                        )
                    )
                else:
                    skip = all(
                        self.nml_ctrl[nml_group].get(cn) == param_dict[cn]
                        for cn in nml_name
                    )
            else:
                print(
                    f"Not all {nml_name} are found in {nml_group}, hence not skipping!"
                )
                skip = False

            if skip:
                print(
                    "-- not creating",
                    expt_path,
                    "- parameters are identical to the control experiment located at",
                    self.base_path,
                    "\n",
                )
                return

        if (
            self.tag_model == "mom6"
        ):  # might need MOM_parameter.all, because many parameters are in-default hence not shown up in `MOM_input`
            # TODO
            pass

    def _parser_mom6_input(self, path):
        """
        Parses MOM6 input file.
        """
        mom6parser = self.MOM6InputParser.MOM6InputParser()
        mom6parser.read_input(path)
        mom6parser.parse_lines()
        return mom6parser

    def _update_config_entries(self, base, change):
        """
        Recursively update nuopc_runconfig and config.yaml entries.
        """
        for k, v in change.items():
            if isinstance(v, dict) and k in base:
                self._update_config_entries(base[k], v)
            else:
                base[k] = v

    def _update_cpl_dt_nuopc_seq(self, seq_path, update_cpl_dt):
        """
        Updates only coupling timestep through nuopc.runseq.
        """
        with open(seq_path, "r") as f:
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
        with open(seq_path, "w") as f:
            f.writelines(update_lines)

    def _get_untracked_files(self, repo):
        """
        Gets untracked git files.
        """
        return repo.untracked_files

    def _get_changed_files(self, repo):
        """
        Gets changed git files.
        """
        return [file.a_path for file in repo.index.diff(None)]

    def _get_deleted_files(self, repo):
        """
        Gets deleted git files.
        """
        return [file.a_path for file in repo.index.diff(None) if file.deleted_file]

    def _restore_swp_files(self, repo, staged_files):
        """
        Restores tmp git files.
        """
        swp_files = [file for file in staged_files if file.endswith(".swp")]
        for file in swp_files:
            repo.git.restore(file)

    def _read_ryaml(self, yaml_path):
        """
        Reads YAML file and preserve comments.
        """
        with open(yaml_path, "r") as f:
            return ryaml.load(f)

    def _write_ryaml(self, data, yaml_path):
        """
        Writes YAML file and preserve comments.
        """
        with open(yaml_path, "w") as f:
            ryaml.dump(data, f)

    def _update_metadata_description(self, metadata, restartpath):
        """
        Updates metadata description with experiment details.
        """
        tmp_string1 = (
            f"\nNOTE: this is a perturbation experiment, but the description above is for the control run."
            f"\nThis perturbation experiment is based on the control run {self.base_path} from {self.base_branch_name}"
        )
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
        """
        Removes comments after the key in metadata.
        """
        if key in metadata:
            metadata.ca.items[key] = [None, None, None, None]

    def _extract_metadata_keywords(self, param_change_dict):
        """
        Extracts keywords from parameter change dictionary.
        """
        keywords = ", ".join(param_change_dict.keys())
        return keywords

    def main(self):
        """
        Main function for the program.
        """
        parser = argparse.ArgumentParser(
            description="Manage ACCESS-OM3 experiments.\
                 Latest version and help: https://github.com/minghangli-uni/Expts_manager"
        )
        parser.add_argument(
            "INPUT_YAML",
            type=str,
            nargs="?",
            default="Expts_manager.yaml",
            help="YAML file specifying parameter values for expt runs. Default is Expts_manager.yaml",
        )
        args = parser.parse_args()
        INPUT_YAML = vars(args)["INPUT_YAML"]

        yamlfile = os.path.join(self.dir_manager, INPUT_YAML)
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
