"""
chapter9 EM algorithm for Gaussian Misture Model
Using iris dataset for clustering

"""
import DatasetUtil as DS
from HTMLTable import HTMLTable
import re
from GMM import GMM


if __name__ == "__main__":
    print("\t============ Chap9 EM for GMM ============")

    ds = DS.DATAUtil()
    x_train, y_train = ds.load(True, r".\dataset.dat")
    model = GMM()
    model.train(x_train)

    y_pred = model.predict(x_train)
    y_train = ds.y_int2str(y_train)

    table = HTMLTable(caption='Iris Data Cluster')
    table.append_header_rows((
        ('No.',    'A1',     'A2',     'A3',     'A4',    'Classification', ''),
        ('',        '',       '',       '',       '',     'Label-C',   'Predict-C'),
    ))
    table[0][0].attr.rowspan = 2
    table[0][1].attr.rowspan = 2
    table[0][2].attr.rowspan = 2
    table[0][3].attr.rowspan = 2
    table[0][4].attr.rowspan = 2
    table[0][5].attr.colspan = 2

    for i in range(x_train.shape[0]):
        table.append_data_rows((
            (f"{i}",
             x_train[i, 0],
             x_train[i, 1],
             x_train[i, 2],
             x_train[i, 3],
             y_train[i],
             str(y_pred[i])),
        ))
    """
    table.append_data_rows((
        ('1', 5.1, 3.5, 1.4, 0.2, 'setosa', 'abc'),
    ))
    table.append_data_rows((
        ('2', 4.9, 3, 1.4, 0.2, 'setosa', 'abc'),
    ))
    """
    table.caption.set_style({
        'font-size': '15px',
    })
    table.set_style({
        'border-collapse': 'collapse',
        'word-break': 'keep-all',
        'white-space': 'nowrap',
        'font-size': '14px',
    })
    table.set_cell_style({
        'border-color': '#000',
        'border-width': '1px',
        'border-style': 'solid',
        'padding': '5px',
        'text-align': 'center',
    })
    table.set_header_row_style({
        'color': '#fff',
        'background-color': '#48a6fb',
        'font-size': '18px',
    })
    table.set_header_cell_style({
        'padding': '15px',
    })
    table[1].set_cell_style({
        'padding': '8px',
        'font-size': '15px',
    })

    html_table = table.to_html()

    with open(r".\output.html", 'r+') as f:
        html = f.read()
        f.seek(0)
        f.truncate()
        pat = re.compile(r'<table(.+?)</table>', flags=re.S)
        html2 = re.sub(pat, html_table, html)
        print(html2)
        f.write(html2)

    pass
