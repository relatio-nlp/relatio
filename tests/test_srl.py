# test for SRL, extract_roles, process_roles + coreference resolution

from relatio.semantic_role_labeling import extract_roles


def test_srl_result_size(df_split, SRL_model):
    num_sentences = df_split.shape[0]
    srl_res = SRL_model(df_split["sentence"], progress_bar=False)
    assert len(srl_res) == num_sentences, "the sizes are incompatible between srl_result list and split_df"


def test_extract_roles_size(srl_res):
    roles, sentence_index = extract_roles(
        srl_res, used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"], progress_bar=False
    )
    assert (
        len(srl_res) == sentence_index[-1] + 1
    ), "the sizes are incompatible between srl_result list and extract_roles size"


def test_post_proc_roles_size(p, roles):
    postproc_roles = p.process_roles(
        roles,
        dict_of_pos_tags_to_keep={
            "ARG0": ["PRON", "NOUN", "PROPN"],
            "B-V": ["VERB"],
            "ARG1": ["NOUN", "PROPN", "PRON"],
        },
        max_length=50,
        progress_bar=False,
        output_path=None,
    )
    assert len(postproc_roles) == len(
        roles
    ), "the sizes are incompatible between extract_roles size and postproc_roles size"


def test_coreference_resolution(df_split, p):
    num_sentences = df_split.shape[0]
    cr_sentences = p.coreference_resolution(df_split["sentence"])
    assert (
        len(cr_sentences) == num_sentences
    ), "the sizes are incompatible between coreference_resolustion_df and split_df"
