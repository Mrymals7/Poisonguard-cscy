# Backdoor Trigger Consistency Audit

## Trigger
Features: ['Bwd IAT Std', 'Fwd Packet Length Std']
Values: {'Bwd IAT Std': 0.022205280624182006, 'Fwd Packet Length Std': 0.014835711945412769}

## Audit Results
- Values within benign ranges: True
- Benign records near both values (tolerance 0.01): 4772
- Validation samples altered: 500
- Other features unchanged: True

## Limitations
The experiment changes two normalized flow features.

Being inside benign feature ranges does not establish
that their combination represents a physically valid
network flow.

Related packet and timing measurements are not
recomputed from raw packet data.

The joint-neighborhood count is exploratory and
does not prove packet-level consistency.

Therefore, these results demonstrate a synthetic
feature-space backdoor, not a verified realistic
network-level attack.

## Experimental Findings
Random Forest mean triggered benign rate across
seeds 7, 42, and 123: 92.2%.

The MLP showed a much smaller backdoor effect.
