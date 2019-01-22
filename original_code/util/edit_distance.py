import numpy as np

class EditDistance(object):
    def __init__(self):
        self.INSERT_COST = 1
        self.DELETE_COST = 1
        self.REPLACE_COST = 1
        self.MATCH_COST = 0
        self.cache = {}
        self.op_cache = {}
        self.saved_str_a = None
        self.saved_str_b = None

    def compute_rec(self, str_a, str_b, i=0, j=0):
        if (i, j) in self.cache:
            return self.cache[(i, j)]

        if i == len(str_a):
            return (len(str_b) - j) * self.DELETE_COST
        if j == len(str_b):
            return (len(str_a) - i) * self.INSERT_COST

        best_score = 99999
        if str_a[i] == str_b[j]:
            # match
            match_score = self.compute_rec(str_a, str_b, i+1, j+1)+self.MATCH_COST
            if best_score > match_score:
                best_score = match_score
                self.op_cache[(i, j)] = 'm'
        else:
            # insert
            insert_score = self.compute_rec(str_a, str_b, i+1, j)+self.INSERT_COST
            if best_score > insert_score:
                best_score = insert_score
                self.op_cache[(i, j)] = 'i'

            # delete
            delete_score = self.compute_rec(str_a, str_b, i, j+1)+self.DELETE_COST
            if best_score > delete_score:
                best_score = delete_score
                self.op_cache[(i, j)] = 'd'

            # replace
            replace_score = self.compute_rec(str_a, str_b, i+1, j+1)+self.REPLACE_COST
            if best_score > replace_score:
                best_score = replace_score
                self.op_cache[(i, j)] = 'r'

        self.cache[(i, j)] = best_score
        return best_score

    # turns str_b into str_a in the minimum number of steps
    def compute(self, str_a, str_b):
        self.cache = {}
        self.op_cache = {}
        self.saved_str_a = str_a
        self.saved_str_b = str_b
        return self.compute_rec(str_a, str_b)

    def operations(self):
        i = 0; j = 0;
        a_len = len(self.saved_str_a)
        b_len = len(self.saved_str_b)
        ops = []
        while True:
            if (i, j) not in self.op_cache:
                # finished mop up the last few characters
                if i == a_len:
                    ops.extend(['d']*(b_len - j))
                if j == b_len:
                    ops.extend(['i']*(a_len - i))
                break

            op = self.op_cache[(i, j)]
            ops.append(op)

            if op == 'm':
                i+=1; j+=1
            elif op == 'i':
                i+=1
            elif op == 'd':
                j+=1
            elif op == 'r':
                i+=1; j+=1
        return "".join(ops)


def main():
    ed = EditDistance()
    print ed.compute("cat", "catr")
    print ed.operations()
    print ed.compute("catr", "cat")
    print ed.operations()
    print ed.compute("cbat", "catr")
    print ed.operations()
    print ed.compute('smtih', 'smith')
    print ed.operations()

    import time

    start_time = time.time()
    for i in xrange(100):
        ed.compute('fsffvfdsbbdfvvdavavavavavava', 'fvdaabavvvvvadvdvavavadfsfsdafvvav')
    end_time = time.time()
    print end_time-start_time


    print ed.operations()
    print ed.compute(list('fsffvfdsbbdfvvdavavavavavava'), list('fvdaabavvvvvadvdvavavadfsfsdafvvav'))
    print ed.operations()

if __name__ == "__main__":
    main()
