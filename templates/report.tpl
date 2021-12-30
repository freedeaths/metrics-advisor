# Report

Hi, this is {{ foo['bar'] }}

## Summary

{% if anomaly != [] %}
    There are {{ anomaly|length }} anomalies.
{% else %}
    Everything runs well.
{% endif %}

## Details

{% for item in sort_corr[:5] %}
* Name: {{ item['name'] }}, Correlation: {{ item['corr']}}
{% endfor %}

{% for item in pics %}
* ![pic]({{ item }})
{% endfor %}
## TBD