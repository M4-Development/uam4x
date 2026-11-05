# Linear system plugin

Solve `A_mat @ x = b` with a fixed square matrix and one-dimensional vectors:

```python
balance = LinearSystemofEqGenerator('balance', solution, num_eq=2)
balance.add_bvec('rhs')
balance.add_Amat([[2, 1], [1, 3]])
balance.add_xvec('forces', tracked=True)
balance.add_to_study()

solution.add_link('loads.vector', balance.input('rhs'))
# Individual elements can also be linked using balance.input('rhs[0]')
# or balance.output('forces[0]').
```

An RHS of `[4, 7]` gives `[1, 2]`. The plugin supplies exact derivatives
with respect to the RHS. `A_mat` is fixed configuration, not a linked input.
Missing generator options, incorrect shapes, and singular solves produce errors.

Run tests with:

```powershell
python uam4x_plugins/general/tests/test_linear_system.py -v
```
