import os,sys,copy
import git,subprocess
import shutil

try:
    import yaml
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True
except ImportError:
    print('\nFatal error: modules not available.')
    print('On NCI, do the following and try again:')
    print('   module use /g/data/hh5/public/modules; module load conda/analysis3\n')
    raise

DIR_MANAGER     = os.getcwd()
DIR_UTILS       = os.path.join(DIR_MANAGER,'tools/om3-utils')
BRANCH_NAME     = '1deg_jra55do_ryf'
BRANCH_NAME_BASE= '_'.join([BRANCH_NAME,'base'])
BRANCH_PERTURB  = 'perturbation'
TEST_REL_PATH   = 'test'
EXPT_REL_PATH   = '/'.join([TEST_REL_PATH,BRANCH_NAME])

try:
    # currently imported from a fork: https://github.com/minghangli-uni/om3-utils,
    # will update the tool when it is merged.
    sys.path.append(DIR_UTILS)
    repo = git.Repo(DIR_UTILS)
    repo.git.checkout('main')  
    from om3utils import MOM6InputParser  
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Failed to import om3utils. Please check the sys.path to ensure the module existing in the directory.")
    raise
# ===============================================


class Expts_manager(object):
    def __init__(self, dir_manager=DIR_MANAGER):
        self.dir_manager = dir_manager
        self.template_path = None
        self.param_dict_change_list = []
        self.param_dict_change_full_list = []
        self.commt_dict_change = {}
        self.startfrom = None

    def setup_template(self, template_yamlfile):
        """Setup the template by 
           reading YAML file,
           checking out the branch,
           enabling metadata in config.yaml of the template,
           loading param_dict, commt_dict
        """
        indata = self._read_ryaml(template_yamlfile)
        template = indata['template']
        self.template_path = os.path.join(self.dir_manager, template)
        repo = git.Repo(self.template_path)
        repo.git.checkout(BRANCH_NAME)
        if BRANCH_NAME_BASE not in repo.branches:
            base_branch = repo.create_head(BRANCH_NAME_BASE)
        else:
            base_branch = repo.branches[BRANCH_NAME_BASE]
        base_branch.checkout()
        print(f"Checkout the base branch: {BRANCH_NAME_BASE}")
        template_config_data = self._read_ryaml(os.path.join(self.template_path,'config.yaml'))
        tmp_template_config_data = copy.deepcopy(template_config_data) # before changed
        template_config_data['metadata']['enable'] = True
        if tmp_template_config_data['metadata']['enable'] != template_config_data['metadata']['enable']:
            self._write_ryaml(os.path.join(self.template_path,'config.yaml'),template_config_data)
            # stage the change
            repo.index.add(os.path.join(self.template_path,'config.yaml'))
            # commmit the change
            commit_message = 'Enable metadata to be true for perturbation tests.'
            repo.index.commit(commit_message)
            print(f"Committed changes with message: '{commit_message}'")
        else:
            print('Nothing changed, hence no further commits to the template repo!')
            
        startfrom = str(indata['startfrom']).strip().lower().zfill(3)
        name_dict = indata['namelists']['MOM_list']
        param_dict = self._parser_mom6_input(os.path.join(self.template_path, 'MOM_input')).param_dict
        commt_dict = self._parser_mom6_input(os.path.join(self.template_path, 'MOM_input')).commt_dict
        self._update_param_dict_change(name_dict, param_dict, commt_dict)

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
            print(expt_path)

            if os.path.exists(expt_path):
                print(' -- not creating ', rel_path, ' - already exists!','\n')
                
            else:
                print(f'clone template - payu clone!','\n')
                # payu clone -B master -b ctrl test/1deg_jra55_ryf test/1deg_jra55_ryf_test
                command = f'payu clone -B {BRANCH_NAME_BASE} -b {BRANCH_PERTURB} {self.template_path} {expt_path}'
                test = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    
                # apply changes and write them to `MOM_override`
                MOM6_or_parser = self._parser_mom6_input(os.path.join(expt_path, 'MOM_override'))
                MOM6_or_parser.param_dict = self.param_dict_change_list[i]
                MOM6_or_parser.commt_dict = self.commt_dict_change
                MOM6_or_parser.writefile_MOM_input(os.path.join(expt_path,'MOM_override'))
                
                # Update config.yaml 
                config_path = os.path.join(expt_path,'config.yaml')
                config_data = self._read_ryaml(config_path)
                config_data['jobname'] = '_'.join([BRANCH_NAME_BASE,expt_name])
                self._write_ryaml(config_path,config_data)
    
                # Update metadata.yaml
                metadata_path = os.path.join(expt_path, 'metadata.yaml')
                metadata = self._read_ryaml(metadata_path)
                self._update_metadata_description(metadata)
                self._remove_metadata_comments('description', metadata)
                keywords = self._extract_metadata_keywords(self.param_dict_change_list[i])
                metadata['keywords'] = f"{BRANCH_NAME_BASE}, {BRANCH_PERTURB}, {keywords}"
                self._remove_metadata_comments('keywords', metadata)
                self._write_ryaml(metadata_path, metadata)

                # replace metadata in archive/
                shutil.copy(metadata_path,os.path.join(expt_path,'archive'))

                # git commit
                # repo = git.Repo(expt_path)
                # print(_get_changed_files_git(repo))
    
                repo = git.Repo(expt_path)
                changed_files = self._get_changed_files(repo)
                untracked_files = self._get_untracked_files(repo)
                files_to_stages = set(changed_files+untracked_files)
                if files_to_stages:
                    print(files_to_stages)
                    repo.index.add(files_to_stages)
                    commit_message = f"Payu clone from the base: {self.template_path}; staged files/directories are: {', '.join(f'{i}' for i in files_to_stages)}"
                    repo.index.commit(commit_message)
                else:
                    print('No files are required to be committed!')

    def _get_untracked_files(self,repo):
        return repo.untracked_files
    
    def _get_changed_files(self,repo):
        return [file.a_path for file in repo.index.diff(None)]    
    
            
    def _read_ryaml(self, yaml_path):
        """ Read yaml file."""
        with open(yaml_path, 'r') as f:
            return ryaml.load(f)
            
    def _write_ryaml(self,yaml_path,data):
        """ Write yaml file."""
        with open(yaml_path, 'w') as f:
            ryaml.dump(data,f)
            
    def _update_metadata_description(self, metadata):
        """Update metadata description with experiment details."""
        desc = (f'\nNOTE: this is a perturbation experiment, but the description above is for the control run.'
                f'\nThis perturbation experiment is based on the control run {self.template_path}')
        if self.startfrom == 'rest':
            desc += '\nbut with condition of rest.'
        metadata['description'] = desc

    def _remove_metadata_comments(self, key, metadata):
        """Remove comments after the key in metadata."""
        if key in metadata:
            metadata.ca.items[key] = [None, None, None, None]

    def _extract_metadata_keywords(self, param_change_dict):
        """Extract keywords from parameter change dictionary."""
        keywords = ', '.join(param_change_dict.keys())
        return keywords

if __name__ == "__main__":
    INPUT_YAML   = './expt_mom.yaml'
    expt_manager = Expts_manager()
    yamlfile     = os.path.join(DIR_MANAGER,INPUT_YAML)
    expt_manager.setup_template(yamlfile)
    expt_manager.manage_expts()

