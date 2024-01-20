# goal programming, preemptive method. first priority is total workers remained idle goal
import case2_data 
import pandas as pd # import pandas
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# Constructing the model as md1
md1 = pyo.ConcreteModel("IE251_CASE2")

# initialize sets
md1.T = pyo.Set(initialize=case2_data.years, doc ="years")
md1.W = pyo.Set(initialize=case2_data.worker_types, doc="worker types")
md1.E = pyo.Set(initialize=case2_data.education_types, doc="education from type C to type B is 0 and education from type B to type A is 1")   
md1.D = pyo.Set(initialize=case2_data.degration_types, doc="degradation from A to B is 1 and from B to C is 0")



#initialize parameters

md1.I = pyo.Param(md1.W, initialize= case2_data.initial_workers, doc="initial workers")
md1.WR = pyo.Param(md1.W, md1.T, initialize= case2_data.workforce_requirements, doc="total workforce requirement per year")
md1.HC = pyo.Param(md1.W,initialize = case2_data.hiring_cost, doc="Hiring cost")
md1.R = pyo.Param(md1.W, initialize= case2_data.resignation_rate, doc="resignation rate")
md1.HY = pyo.Param(md1.W, initialize= case2_data.hiring_limit, doc="hiring limit")
md1.EC = pyo.Param(md1.E, initialize = case2_data.education_cost, doc="education_cost")
md1.IC = pyo.Param(md1.W, initialize= case2_data.idleness_cost, doc="idleness cost")
md1.OC = pyo.Param(md1.W, initialize= case2_data.outsource_cost, doc="outsourcing cost")
md1.PTC = pyo.Param(md1.W, initialize= case2_data.parttime_cost, doc="cost for part time workers")
md1.ED = pyo.Param(md1.E, initialize=case2_data.education_limit, doc="education limit ")
md1.ER = pyo.Param(initialize = 0.5, doc="resignation rate of degraded workers")
md1.PT = pyo.Param(initialize = 80, doc="part time worker limit per year")
md1.OR = pyo.Param(initialize = 175, doc="outsource worker limit per year")



# define decision variables

md1.H = pyo.Var(md1.W, md1.T, doc="number of new workers type w hired in year t", within=pyo.NonNegativeReals)
md1.DW = pyo.Var(md1.D, md1.T, doc="number of degradation type d in year t", within=pyo.NonNegativeReals)
md1.IW = pyo.Var(md1.W, md1.T, doc="number of worker type w remained idle in year t", within=pyo.NonNegativeReals)
md1.OW = pyo.Var(md1.W, md1.T, doc="number of workers type w outsourced in year t", within=pyo.NonNegativeReals)
md1.PTW = pyo.Var(md1.W, md1.T, doc="number of worker type w work part-time in year t", within=pyo.NonNegativeReals)
md1.EW = pyo.Var(md1.E, md1.T, doc="number of education type e in year t",within=pyo.NonNegativeReals)
md1.dv1_plus = pyo.Var(doc="deviation of total cost from the goal(excess) ", within=pyo.NonNegativeReals)
md1.dv1_minus = pyo.Var(doc="deviation of total cost from the goal(slack)", within=pyo.NonNegativeReals)
md1.dv2_plus = pyo.Var(doc="deviation of total idle from the goal(excess)", within=pyo.NonNegativeReals)
md1.dv2_minus = pyo.Var(doc="deviation of total idle from the goal(slack)", within=pyo.NonNegativeReals)


# set constraints

def hiring_limit_rule(md1, w, t):         
    return md1.H[w,t] <= md1.HY[w]
md1.hiring_limit = pyo.Constraint(md1.W, md1.T, rule=hiring_limit_rule, doc="hiring limit of worker type w constraint")

def outsource_limit_rule(md1, t):             
    return sum(md1.OW[w, t] for w in md1.W) <= md1.OR
md1.outsource_limit = pyo.Constraint(md1.T, rule=outsource_limit_rule, doc="# outsourced worker per year limit constraint ")

def parttime_limit_rule(md1, t):               
    return sum(md1.PTW[w, t] for w in md1.W) <= md1.PT
md1.parttime_limit = pyo.Constraint(md1.T, rule=parttime_limit_rule, doc="parttime worker per year limit constraint")

def workforce_requirement_A_2024_rule(md1):   
    return md1.I[0] +  md1.H[0,0] - md1.IW[0,0] + md1.OW[0,0] + (md1.PTW[0,0]/2) + md1.EW[1,0] - md1.DW[1,0] == md1.WR[0,0]
md1.workforce_requirement_A_2024 = pyo.Constraint(rule = workforce_requirement_A_2024_rule, doc=" workforce requirement of type A in 2024 constraint")


def workforce_requirement_B_2024_rule(md1):     
    return md1.I[1] +  md1.H[1,0] - md1.IW[1,0] + md1.OW[1,0] + (md1.PTW[1,0] /2) - md1.EW[1,0] + md1.EW[0,0] + (md1.DW[1,0] * md1.ER) - md1.DW[0,0] == md1.WR[1,0]
md1.workforce_requirement_B_2024 = pyo.Constraint(rule = workforce_requirement_B_2024_rule, doc=" workforce requirement of type B in 2024 constraint")

def workforce_requirement_C_2024_rule(md1):         
    return md1.I[2] +  md1.H[2,0] - md1.IW[2,0] + md1.OW[2,0] + (md1.PTW[2,0] /2) - md1.EW[0,0] +( md1.DW[0,0] * md1.ER) == md1.WR[2,0]
md1.workforce_requirement_C_2024 = pyo.Constraint(rule = workforce_requirement_C_2024_rule, doc=" workforce requirement of type C in 2024 constraint")


def total_workers_A_end_2024_rule(md1):           
    return md1.I[0] + md1.H[0,0] + md1.EW[1,0] - md1.DW[1,0]
md1.total_workers_A_end_2024 = pyo.Expression(rule = total_workers_A_end_2024_rule, doc="total worker of type A at the end of 2024 expression")

def total_workers_B_end_2024_rule(md1):          
    return md1.I[1] + md1.H[1,0] + md1.EW[0,0] - md1.EW[1,0] + (md1.DW[1,0] * md1.ER) - md1.DW[0,0]
md1.total_workers_B_end_2024 = pyo.Expression(rule = total_workers_B_end_2024_rule, doc="total worker of type B at the end of 2024 expression")

def total_workers_C_end_2024_rule(md1):
    return md1.I[2] + md1.H[2,0] - md1.EW[0,0] + ( md1.DW[0,0] * md1.ER)  
md1.total_workers_C_end_2024 = pyo.Expression(rule = total_workers_C_end_2024_rule, doc="total worker of type C at the end of 2024 expression")





def workforce_requirement_A_year_2025_rule(md1): 
    return md1.total_workers_A_end_2024 - (md1.total_workers_A_end_2024 * md1.R[0]) + md1.H[0,1] + md1.EW[1,1] - md1.DW[1,1] - md1.IW[0,1] + md1.OW[0,1] + (md1.PTW[0,1] /2 )  == md1.WR[0,1]
md1.workforce_requirement_A_2025 = pyo.Constraint(rule = workforce_requirement_A_year_2025_rule, doc=" workforce requirement of type A in 2025 constraint")

def total_workers_at_the_end_of_2025_A_rule(md1):
    return md1.total_workers_A_end_2024 - (md1.total_workers_A_end_2024 * md1.R[0]) + md1.H[0,1] + md1.EW[1,1] - md1.DW[1,1]
md1.total_workers_at_the_end_of_2025_A = pyo.Expression(rule = total_workers_at_the_end_of_2025_A_rule , doc="total worker of type A at the end of 2025 expression")



def workforce_requirement_B_2025_rule(md1):     
    return md1.total_workers_B_end_2024 - (md1.total_workers_B_end_2024 * md1.R[1]) + md1.H[1,1] - md1.EW[1,1] + md1.EW[0,1] +( md1.DW[1,1] * md1.ER) - md1.DW[0,1] - md1.IW[1,1] + md1.OW[1,1] + (md1.PTW[1,1] /2)  == md1.WR[1,1]
md1.workforce_requirement_B_2025 = pyo.Constraint( rule= workforce_requirement_B_2025_rule, doc=" workforce requirement of type B in 2025 constraint")

def total_workers_at_the_end_of_2025_B_rule(md1):
    return  md1.total_workers_B_end_2024 - (md1.total_workers_B_end_2024 * md1.R[1]) + md1.H[1,1] - md1.EW[1,1] + md1.EW[0,1] + (md1.DW[1,1] * md1.ER) - md1.DW[0,1]
md1.total_workers_at_the_end_of_2025_B = pyo.Expression(rule = total_workers_at_the_end_of_2025_B_rule , doc="total worker of type B at the end of 2025 expression")






def workforce_requirement_C_2025_rule(md1):     
   return md1.total_workers_C_end_2024 - (md1.total_workers_C_end_2024 * md1.R[2]) + md1.H[2,1] - md1.EW[0,1] + (md1.DW[0,1] * md1.ER) - md1.IW[2,1] + md1.OW[2,1] + (md1.PTW[2,1] /2)  == md1.WR[2,1]
md1.workforce_requirement_C_2025 = pyo.Constraint( rule= workforce_requirement_C_2025_rule, doc=" workforce requirement of type C in 2025 constraint")  


def total_workers_at_the_end_of_2025_C_rule(md1):
    return md1.total_workers_C_end_2024 - (md1.total_workers_C_end_2024 * md1.R[2]) + md1.H[2,1] - md1.EW[0,1] + (md1.DW[0,1] * md1.ER)
md1.total_workers_at_the_end_of_2025_C = pyo.Expression(rule = total_workers_at_the_end_of_2025_C_rule , doc="total worker of type C at the end of 2025 expression")







def workforce_requirement_A_2026_rule(md1):
    return  md1.total_workers_at_the_end_of_2025_A - (md1.total_workers_at_the_end_of_2025_A * md1.R[0]) + md1.H[0,2] + md1.EW[1,2] - md1.DW[1,2] - md1.IW[0,2] + md1.OW[0,2] +( md1.PTW[0,2] /2) == md1.WR[0,2]
md1.workforce_requirement_A_2026 = pyo.Constraint( rule = workforce_requirement_A_2026_rule, doc=" workforce requirement of type A in 2026 constraint")





def workforce_requirement_B_2026_rule(md1):
    return  md1.total_workers_at_the_end_of_2025_B - (md1.total_workers_at_the_end_of_2025_B * md1.R[1]) + md1.H[1,2] - md1.EW[1,2] + md1.EW[0,2] +( md1.DW[1,2] * md1.ER) - md1.DW[0,2] - md1.IW[1,2] + md1.OW[1,2] + (md1.PTW[1,2] /2) == md1.WR[1,2]
md1.workforce_requirement_B_2026 = pyo.Constraint( rule= workforce_requirement_B_2026_rule, doc=" workforce requirement of type B in 2026 constraint")






def workforce_requirement_C_2026_rule(md1):
    return md1.total_workers_at_the_end_of_2025_C - (md1.total_workers_at_the_end_of_2025_C * md1.R[2]) + md1.H[2,2] - md1.EW[0,2] + (md1.DW[0,2] * md1.ER) - md1.IW[2,2] + md1.OW[2,2] + (md1.PTW[2,2] /2)  == md1.WR[2,2]
md1.workforce_requirement_C_2026 = pyo.Constraint( rule= workforce_requirement_C_2026_rule, doc=" workforce requirement of type C in 2026 constraint")


def education_limit_C_to_B_rule(md1,t):
    return md1.EW[0, t] <= md1.ED[0]
md1.education_limit_C_to_B = pyo.Constraint(md1.T, rule = education_limit_C_to_B_rule, doc="education limit from type C to type B per year constraint")


def education_limit_B_to_A_2024_rule(md1): 
    return md1.EW[1,0] <= md1.I[1] * md1.ED[1]
md1.education_limit_B_to_A_2025 = pyo.Constraint(rule= education_limit_B_to_A_2024_rule, doc="education limit from type B to type A in 2024 constraint")




def education_limit_B_to_A_2025_rule(md1):
    return md1.EW[1,1] <= md1.total_workers_B_end_2024 * md1.ED[1]
md1.education_limit_B_to_A_2025 = pyo.Constraint(rule= education_limit_B_to_A_2025_rule, doc="education limit from type B to type A in 2025 constraint")

def education_limit_B_to_A_2026_rule(md1):
    return md1.EW[1,2] <= md1.total_workers_at_the_end_of_2025_B * md1.ED[1]
md1.education_limit_B_to_A_2026 = pyo.Constraint(rule= education_limit_B_to_A_2026_rule, doc="education limit from type B to type A in 2026 constraint")


def total_cost_rule(md1):
    hiring_cost = sum(md1.H[w,t] * md1.HC[w] for w in md1.W for t in md1.T)
    idleness_cost = sum(md1.IW[w,t] * md1.IC[w] for w in md1.W for t in md1.T) 
    outsourcing_cost = sum(md1.OW[w,t] * md1.OC[w] for w in md1.W for t in md1.T)
    parttime_cost = sum(md1.PTW[w,t] * md1.PTC[w] for w in md1.W for t in md1.T)
    education_cost = sum(md1.EW[e,t] * md1.EC[e] for e in md1.E for t in md1.T)
    
    return hiring_cost + idleness_cost + outsourcing_cost + parttime_cost + education_cost
md1.total_cost = pyo.Expression(rule = total_cost_rule , doc="Total cost")

def idle_workers_rule(md1):
    return sum(md1.IW[w,t] for w in md1.W for t in md1.T)
md1.idle_workers = pyo.Expression(rule = idle_workers_rule, doc="total number of workers remained idle")

def idle_workers_constraint_rule(md1):
    return md1.idle_workers + md1.dv2_minus - md1.dv2_plus == 2200    # total number of workers remained idle limit
md1.idle_workers_constraint = pyo.Constraint(rule = idle_workers_constraint_rule, doc=" total number of workers remained idle goal constraint")

#def main_objective_rule(md1):

   # return  md1.dv2_plus     # first objective to get the minimum value of dv2_plus
#md1.main_objective = pyo.Objective(rule = main_objective_rule, sense=pyo.minimize,doc="Objective value")


def dv2plus_constraint_rule(md1):
    return md1.dv2_plus == 671
md1.dv2plus_constraint = pyo.Constraint(rule = dv2plus_constraint_rule)






def total_cost_constraint_rule(md1):
    
    return md1.total_cost  + md1.dv1_minus - md1.dv1_plus == 3200000    # total cost limit
md1.total_cost_constraint = pyo.Constraint(rule = total_cost_constraint_rule, doc="total cost goal constraint")

def main_objective_rule(md1):
    return md1.dv1_plus
md1.main_objective = pyo.Objective(rule = main_objective_rule, sense= pyo.minimize )






md1.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #shadow prices of the constraints
md1.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT) #reduced costs of the objective function coefficients

Solver = SolverFactory('glpk')  # assign glpk as solver

md1.write('mdl_labels.lp', io_options={'symbolic_solver_labels': True})   
md1.write('mdl_nolabels.lp', io_options={'symbolic_solver_labels': False}) 
Solver.options['ranges']= r"YOURFILEPATH\SA_Report.txt"
md1.write('md1.lp', io_options={'symbolic_solver_labels': True})  # open form of the model
md1.write('md1.lp', io_options={'symbolic_solver_labels': False})

SolverResults = Solver.solve(md1, tee=True)  # solve the model


SolverResults.write()  # write solver  results
md1.pprint()
md1.total_cost.display() # show value of objective function in console
md1.dv1_plus.display()
md1.dv2_plus.display()
md1.dv1_minus.display()

hired_worker_dict = md1.H.extract_values()   # get hired worker values
degraded_worker_dict = md1.DW.extract_values()  # get degraded worker values
idle_workers_dict = md1.IW.extract_values()     # get  worker remained idle values
outsourced_workers_dict = md1.OW.extract_values()   # get outsourced worker values
parttime_workers_dict = md1.PTW.extract_values()    # get part-time worker values
educated_workers_dict = md1.EW.extract_values()     # get education values
dv1_plus_dict = md1.dv1_plus.extract_values()
dv1_minus_dict = md1.dv1_minus.extract_values()
dv2_plus_dict = md1.dv2_plus.extract_values()
dv2_minus_dict = md1.dv2_minus.extract_values()


optimal = {0: pyo.value(md1.total_cost)}
optimalvalue = pd.DataFrame.from_dict(optimal,orient="index", columns=["optimal value"] ) 
optimalvalue.to_excel(r"YOURFILEPATH\optimalvalue.xlsx")    # write total cost value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.H.items()}
optimal_hired_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for hired workers"])
optimal_hired_workers.to_excel(r'YOURFILEPATH\hired.xlsx', sheet_name='Hired workers')    # write hired worker value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.DW.items()}
optimal_degraded_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for degraded workers"])
optimal_degraded_workers.to_excel(r'YOURFILEPATH\degraded.xlsx', sheet_name='degraded workers')    #  write degraded worker value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.IW.items()}
optimal_idle_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for idle workers"])
optimal_idle_workers.to_excel(r'YOURFILEPATH\idle.xlsx', sheet_name='idle workers')   # write worker remained idle value to excel


variable_data = {(i, v.name): pyo.value(v) for i,v in md1.OW.items()}
optimal_outsourced_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for outsourced workers"])
optimal_outsourced_workers.to_excel(r'YOURFILEPATH\outsourced.xlsx', sheet_name='outsourced workers')   # write outsourced worker value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.PTW.items()}
optimal_parttime_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for parttime workers"])
optimal_parttime_workers.to_excel(r'YOURFILEPATH\parttime.xlsx', sheet_name='parttime workers')   # write part time worker value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.EW.items()}
optimal_educated_workers = pd.DataFrame.from_dict(variable_data, orient="index", columns=["variable value for educated workers"])
optimal_educated_workers.to_excel(r'YOURFILEPATH\educated.xlsx', sheet_name='educated workers')    # write education value to excel

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.dv1_plus.items()}
optimal_dv1_plus = pd.DataFrame.from_dict(variable_data, orient="index", columns=["dv1plus"])
optimal_dv1_plus.to_excel(r'YOURFILEPATH\dv1plus.xlsx', sheet_name='dv1plus')


variable_data = {(i, v.name): pyo.value(v) for i,v in md1.dv1_minus.items()}
optimal_dv1_minus = pd.DataFrame.from_dict(variable_data, orient="index", columns=["dv1minus"])
optimal_dv1_minus.to_excel(r'YOURFILEPATH\dv1minus.xlsx', sheet_name='dv1minus')

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.dv2_plus.items()}
optimal_dv2_plus = pd.DataFrame.from_dict(variable_data, orient="index", columns=["dv2plus"])
optimal_dv2_plus.to_excel(r'YOURFILEPATH\dv2plus.xlsx', sheet_name='dv2plus')

variable_data = {(i, v.name): pyo.value(v) for i,v in md1.dv1_plus.items()}
optimal_dv2_minus = pd.DataFrame.from_dict(variable_data, orient="index", columns=["dv2minus"])
optimal_dv2_minus.to_excel(r'YOURFILEPATH\dv2minus.xlsx', sheet_name='dv2minus')


#export to excel: reduced costs
reduced_cost_dict={str(key):md1.rc[key] for key in md1.rc.keys()}
Reduced_Costs_print =pd.DataFrame.from_dict(reduced_cost_dict,orient="index", columns=["reduced cost"])
Reduced_Costs_print.to_excel(r'YOURFILEPATH\ReducedCostsPart1.xlsx', sheet_name='ReducedCosts')



#export to excel: shadow prices        
duals_dict = {str(key):md1.dual[key] for key in md1.dual.keys()}

u_slack_dict = {
    # uslacks for non-indexed constraints
    **{str(con):con.uslack() for con in md1.component_objects(pyo.Constraint)
       if not con.is_indexed()},
    # indexed constraint uslack
    # loop through the indexed constraints
    # get all the indices then retrieve the slacks for each index of constraint
    **{k:v for con in md1.component_objects(pyo.Constraint) if con.is_indexed()
       for k,v in {'{}[{}]'.format(str(con),key):con[key].uslack()
                   for key in con.keys()}.items()}
    }

l_slack_dict = {
    # lslacks for non-indexed constraints
    **{str(con):con.lslack() for con in md1.component_objects(pyo.Constraint)
       if not con.is_indexed()},
    # indexed constraint lslack
    # loop through the indexed constraints
    # get all the indices then retrieve the slacks for each index of constraint
    **{k:v for con in md1.component_objects(pyo.Constraint) if con.is_indexed()
       for k,v in {'{}[{}]'.format(str(con),key):con[key].lslack()
                   for key in con.keys()}.items()}
    }

# combine into a single df
Shadow_Prices_print = pd.concat([pd.Series(d,name=name)
           for name,d in {'duals':duals_dict,
                          'uslack':u_slack_dict,
                          'lslack':l_slack_dict}.items()],
          axis='columns')
Shadow_Prices_print.to_excel(r'YOURFILEPATH\ShadowPricesPart1.xlsx', sheet_name='ShadowPrices')



import pyomo_sens_analysis_v2 as pyo_SA 
pyo_SA.reorganize_SA_report(file_path_SA = r"YOURFILEPATH\SA_Report.txt", 
                            file_path_LP_labels = r"YOURFILEPATH\mdl_labels.lp", 
                                file_path_LP_nolabels = r"YOURFILEPATH\mdl_nolabels.lp")






























