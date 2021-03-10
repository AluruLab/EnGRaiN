import sys
in_file="BIOGRID.csv"
#in_file="test.csv"
with open(in_file) as bf:
    hx = bf.readline()
    for line in bf:
        elts = line.strip().split("\t")
        try:
            inter_a = elts[0].replace('"', '').split("|")
            inter_b = elts[1].replace('"', '').split("|")
            if len(inter_a) > 1:
                alias_a = inter_a[1].split(":")[1]
            else:
                alias_a = "-"
            if len(inter_a) > 2:
                gene_a = inter_a[2].split(":")[1]
            else:
                gene_a = "-"
            if len(inter_b) > 1:
                alias_b = inter_b[1].split(":")[1]
            else:
                alias_b = "-"
            if len(inter_b) > 2:
                gene_b = inter_b[2].split(":")[1]
            else:
                gene_b = "-"
            gene_syn_a = elts[2].replace('"', '')
            if gene_syn_a == "-":
                syn_names_a = "-"
            else:
               syn_lst_a = [x.split(":")[1].replace("(gene name synonym)", "") for x in gene_syn_a.split("|")]
               syn_names_a = ";".join(syn_lst_a)
            gene_syn_b = elts[3].replace('"', '')
            if gene_syn_b == "-":
                syn_names_b = "-"
            else:
                syn_lst_b = [x.split(":")[1].replace("(gene name synonym)", "") for x in gene_syn_b.split("|")]
                syn_names_b = ";".join(syn_lst_b)
            detect_method = elts[4].replace('"', '').split('(')[1].replace(')','')
            int_type = elts[7].replace('"', '').split('(')[1].replace(')','')
            oline = [alias_a, gene_a, alias_b, gene_b, syn_names_a, syn_names_b, detect_method, int_type]
            print("\t".join(oline))
        except IndexError as ex:
            print(elts, file=sys.stderr)
            print(ex, file=sys.stderr)
            exit(1)
