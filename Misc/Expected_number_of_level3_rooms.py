import numpy as np

NR_ROOMS = 4
NR_TEMPLES_TO_FULL = 12 // NR_ROOMS

NR_TEMPLES_AVAIL = 50
NR_FULL_TEMPLES = NR_TEMPLES_AVAIL // NR_TEMPLES_TO_FULL

NR_SAMPLES = 10000

max_tier_rooms = np.zeros((NR_SAMPLES, 1))

def upgrade_room(tier: int):
    if tier == 0:
        return 1
    elif tier == 1:
        rnd = np.random.uniform()
        if rnd > 0.6:
            return 3
        else:
            return 2
    return 3

for sample_id in range(NR_SAMPLES):
    avg_max_tier = 0

    for _ in range(NR_FULL_TEMPLES):
        room_tiers = np.zeros((11, ))
        indices = np.random.choice(list(range(11)), size=(6,), replace=False)
        room_tiers[indices] = 1

        rooms_not_max_tier = list(range(11))

        for _ in range(NR_TEMPLES_TO_FULL):
            upgrade_indices = np.random.choice(rooms_not_max_tier, size=(NR_ROOMS, ), replace=False)
            for idx in upgrade_indices:
                upgraded_tier = upgrade_room(room_tiers[idx])
                if upgraded_tier == 3:
                    rooms_not_max_tier.remove(idx)
                room_tiers[idx] = upgraded_tier
        
        avg_max_tier += np.sum(room_tiers == 3)
    
    max_tier_rooms[sample_id] = avg_max_tier / NR_FULL_TEMPLES

print(np.mean(max_tier_rooms))
print(np.var(max_tier_rooms))

