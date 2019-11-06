"""
FROM: https://github.com/dfcf93/MSMARCOV2/blob/master/Ranking/Evaluation/msmarco_eval.py

-------------------


This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 07/05/2018
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import sys
import statistics
import copy
from collections import Counter


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split()
            qid = l[0]
            if int(l[3]) == 0:
                continue
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(l[2].strip()) # changed here from original for new qrel format
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_candidate_from_stream(f,space_for_rank=1000):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split()

            if len(l) == 4: # own format
                qid = l[0]
                pid = l[1].strip()
                rank = int(l[2])
            if len(l) == 6: # original trec format
                qid = l[0]
                pid = l[2].strip()
                rank = int(l[3])
            if qid in qid_to_ranked_candidate_passages:
                pass    
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * space_for_rank
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1]=pid
        except:
            print("error for ",l)
            #raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages

def load_candidate_from_stream_with_score(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split()
            qid = l[0]
            pid = l[1].strip()
            rank = int(l[2])
            score = float(l[3])
            if qid not in qid_to_ranked_candidate_passages:
                qid_to_ranked_candidate_passages[qid] = []
            qid_to_ranked_candidate_passages[qid].append((pid,rank,score))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages

def load_candidate(path_to_candidate,space_for_rank=1000):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    
    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f,space_for_rank)
    return qid_to_ranked_candidate_passages

def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check whether the queries match the evaluation set
    if candidate_set != ref_set:
        if candidate_set >= ref_set:
            # This is to be expected, since we have split the evaluation set in validation & test
            pass
        elif candidate_set < ref_set:
            message = "Not all queries seem to be ranked. Are you scoring the right set?"
        else:
            message = "The submitted queries do not fully match the queries in the evaluation set. Are you scoring the right set?"

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,MaxMRRRank = 10):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0,MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1/(i + 1)
                    ranking.pop()
                    ranking.append(i+1)
                    break

    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR/len(ranking)
    all_scores['MRR'] = MRR
    all_scores['QueriesRanked'] = len(ranking)
    all_scores['QueriesWithNoRelevant'] = sum((1 for x in ranking if x == 0))
    all_scores['QueriesWithRelevant'] = sum((1 for x in ranking if x > 0))

    all_scores['AverageRankGoldLabel@'+str(MaxMRRRank)] = statistics.mean((x for x in ranking if x > 0))
    all_scores['MedianRankGoldLabel@'+str(MaxMRRRank)] = statistics.median((x for x in ranking if x > 0))

    all_scores['AverageRankGoldLabel'] = statistics.mean(ranking)
    all_scores['MedianRankGoldLabel'] = statistics.median(ranking)
    all_scores['HarmonicMeanRankingGoldLabel'] = statistics.harmonic_mean(ranking)
    return all_scores
                
def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

def compute_metrics_with_cs_at_n(path_to_reference,path_to_neural_output,candidate_set,candidate_from_to,perform_checks=True):
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_neural_output)
    return compute_metrics_with_cs_at_n_memory(qids_to_relevant_passageids,qids_to_ranked_candidate_passages,candidate_set,candidate_from_to,perform_checks)

def compute_metrics_with_cs_at_n_memory(qids_to_relevant_passageids,qids_to_ranked_candidate_passages,candidate_set,candidate_from_to,perform_checks=True):

    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)
    
    if type(candidate_from_to) != int:

        results = {}
        for i in range(candidate_from_to[0], candidate_from_to[1] + 1):

            pruned_qids_to_ranked_candidate_passages = {}

            for query,rank_list in qids_to_ranked_candidate_passages.items(): #qid_to_ranked_candidate_passages[qid][rank-1]=pid
                pruned_qids_to_ranked_candidate_passages[query] = [0] * 1000
                added = 0
                for rank, pid in enumerate(rank_list):
                    if pid == 0: # 0 means no more entries > 0 
                        break

                    #rank = rank + 1
                    if pid in candidate_set[query] and candidate_set[query][pid] <= i:
                        pruned_qids_to_ranked_candidate_passages[query][added] = pid
                        added += 1


            results[i] = compute_metrics(qids_to_relevant_passageids, pruned_qids_to_ranked_candidate_passages)

        return results

    else: # assume single number
        pruned_qids_to_ranked_candidate_passages = {}

        for query,rank_list in qids_to_ranked_candidate_passages.items(): #qid_to_ranked_candidate_passages[qid][rank-1]=pid
            pruned_qids_to_ranked_candidate_passages[query] = [0] * 1000
            added = 0
            for rank, pid in enumerate(rank_list):
                if pid == 0: # 0 means no more entries > 0 
                    break
                if pid in candidate_set[query] and candidate_set[query][pid] <= candidate_from_to:
                    pruned_qids_to_ranked_candidate_passages[query][added] = pid
                    added += 1

        return compute_metrics(qids_to_relevant_passageids, pruned_qids_to_ranked_candidate_passages)


def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    
    or: 
    
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file> <path_to_query-id_subset_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')

    elif len(sys.argv) == 4:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        path_to_query_to_select = sys.argv[3]

        qids_to_relevant_passageids = load_reference(path_to_reference)
        qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)

        select_queries = {}
        with open(path_to_query_to_select,'r') as f:
            for line in f:
                select_queries[int(line.split("\t")[0])] = 1

        for q in list(qids_to_ranked_candidate_passages.keys()):
            if q not in select_queries:
                qids_to_ranked_candidate_passages.pop(q,None)
                    
        metrics = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')
    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        exit()
    
if __name__ == '__main__':
    main()
