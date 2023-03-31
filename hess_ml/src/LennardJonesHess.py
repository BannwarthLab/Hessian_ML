

import numpy as np


def alp_a_alp_a(C12,C6,a,b):
    x_a = a[0]
    y_a = a[1]
    z_a = a[2]

    x_b = b[0]
    y_b = b[1]
    z_b = b[2]

    res = (6*C6)/((x_a-x_b)**2+(z_a-z_b)**2+(y_a-y_b)**2)**4 - (48*C6*(x_a-x_b)**2)/((x_a-x_b)**2+(z_a-z_b)**2+(y_a-y_b)**2)**5-(12*C12)/((x_a-x_b)**2+(z_a-z_b)**2+(y_a-y_b)**2)**7+(168*C12*(x_a-x_b)**2)/((x_a-x_b)**2+(z_a-z_b)**2+(y_a-y_b)**2)**8

    return res


def alp_a_alp_b(C12,C6,a,b):
    x_a = a[0]
    y_a = a[1]
    z_a = a[2]

    x_b = b[0]
    y_b = b[1]
    z_b = b[2]

    res  = (48*D*(x_a-x_b)^2)/((x_a-x_b)^2+(z_a-z_b)^2+(y_a-y_b)^2)^5-(168*C*(x_a-x_b)^2)/((x_a-x_b)^2+(z_a-z_b)^2+(y_a-y_b)^2)^8-(6*D)/((x_a-x_b)^2+(z_a-z_b)^2+(y_a-y_b)^2)^4+(12*C)/((x_a-x_b)^2+(z_a-z_b)^2+(y_a-y_b)^2)^7

    return res

def alp_a_beta_a(C12,C6,a,b):
    x_a = a[0]
    y_a = a[1]
    z_a = a[2]

    x_b = b[0]
    y_b = b[1]
    z_b = b[2]

    res  =  (168*C*(x_a-x_b)*(y_a-y_b))/((y_a-y_b)^2+(z_a-z_b)^2+(x_a-x_b)^2)^8-(48*D*(x_a-x_b)*(y_a-y_b))/((y_a-y_b)^2+(z_a-z_b)^2+(x_a-x_b)^2)^5

    return res


def alp_a_beta_b(C12,C6,a,b):
    x_a = a[0]
    y_a = a[1]
    z_a = a[2]

    x_b = b[0]
    y_b = b[1]
    z_b = b[2]

    res  =  (168*C*(x_a-x_b)*(y_a-y_b))/((y_a-y_b)^2+(z_a-z_b)^2+(x_a-x_b)^2)^8-(48*D*(x_a-x_b)*(y_a-y_b))/((y_a-y_b)^2+(z_a-z_b)^2+(x_a-x_b)^2)^5

    return res
   